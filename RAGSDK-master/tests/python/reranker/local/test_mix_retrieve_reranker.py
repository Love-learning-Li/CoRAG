#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest

from langchain_core.documents import Document

from mx_rag.reranker.local import MixRetrieveReranker


class TestMixRetrieveReranker(unittest.TestCase):

    def test_min_max_normalize_empty(self):
        result = MixRetrieveReranker._min_max_normalize([])
        self.assertEqual(result, [])

    def test_min_max_normalize_constant(self):
        result = MixRetrieveReranker._min_max_normalize([5.0, 5.0, 5.0])
        self.assertEqual(result, [0.0, 0.0, 0.0])

    def test_min_max_normalize_normal(self):
        scores = [1, 2, 3, 4, 5]
        result = MixRetrieveReranker._min_max_normalize(scores)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for a, b in zip(result, expected):
            self.assertAlmostEqual(a, b, delta=1e-5)

    def test_normalize_retrieval_results(self):
        dense_docs = [
            self._create_doc("d1", 0.9, "dense"),
            self._create_doc("d2", 0.3, "dense"),
        ]
        sparse_docs = [
            self._create_doc("s1", 10.0, "sparse"),
            self._create_doc("s2", 2.0, "sparse"),
        ]

        MixRetrieveReranker._normalize_retrieval_results(dense_docs, sparse_docs)

        # dense: [0.9, 0.3] -> norm: [1.0, 0.0]
        self.assertEqual(dense_docs[0].metadata["norm_score"], 1.0)
        self.assertEqual(dense_docs[1].metadata["norm_score"], 0.0)

        # sparse: [10.0, 2.0] -> norm: [1.0, 0.0]
        self.assertEqual(sparse_docs[0].metadata["norm_score"], 1.0)
        self.assertEqual(sparse_docs[1].metadata["norm_score"], 0.0)

    def test_logistic_func(self):
        reranker = MixRetrieveReranker(slope=1.0, midpoint=6, baseline=0.4, amplitude=0.3)

        w1 = reranker._logistic_func(1)
        w5 = reranker._logistic_func(5)
        w10 = reranker._logistic_func(10)

        # 短查询权重接近 0.4，长查询权重接近 0.7
        self.assertAlmostEqual(w1, 0.4, delta=0.05)
        self.assertAlmostEqual(w10, 0.7, delta=0.05)

        # 单调递增趋势
        self.assertGreater(w10, w5)
        self.assertGreater(w5, w1)

    def test_rerank_weighted_merge_and_sort(self):
        reranker = MixRetrieveReranker(k=3)

        # 构造测试文档
        docs = [
            self._create_doc("dense_high", 0.95, "dense"),
            self._create_doc("dense_low", 0.10, "dense"),
            self._create_doc("sparse_high", 100.0, "sparse"),
            self._create_doc("sparse_low", 10.0, "sparse"),
            self._create_doc("both_part", 0.80, "dense"),
            self._create_doc("both_part", 0.80, "sparse"),
        ]

        result = reranker.rerank(query="long query for high dense weight", texts=docs)

        self.assertEqual(len(result), 3)

        # 验证 both_part 被正确合并，且 retrieval_type 变为 'both'
        both_doc = next((doc for doc in result if doc.page_content == "both_part"), None)
        self.assertIsNotNone(both_doc)
        self.assertEqual(both_doc.metadata["retrieval_type"], "both")

        # 验证 dense_high 在结果前两名
        self.assertIn("dense_high", [doc.page_content for doc in result[:2]])

    def test_rerank_only_dense_or_sparse(self):
        reranker = MixRetrieveReranker(k=2)

        # 只有 dense
        dense_only = [
            self._create_doc("d1", 0.9, "dense"),
            self._create_doc("d2", 0.1, "dense"),
        ]
        result_dense = reranker.rerank(query="test", texts=dense_only)
        self.assertEqual(len(result_dense), 2)
        self.assertEqual(result_dense[0].page_content, "d1")

        # 只有 sparse
        sparse_only = [
            self._create_doc("s1", 100.0, "sparse"),
            self._create_doc("s2", 10.0, "sparse"),
        ]
        result_sparse = reranker.rerank(query="test", texts=sparse_only)
        self.assertEqual(len(result_sparse), 2)
        self.assertEqual(result_sparse[0].page_content, "s1")

    # 辅助函数：创建测试 Document
    def _create_doc(self, content: str, score: float, retrieval_type: str):
        return Document(
            page_content=content,
            metadata={
                "score": score,
                "retrieval_type": retrieval_type,
            }
        )


if __name__ == '__main__':
    unittest.main()
