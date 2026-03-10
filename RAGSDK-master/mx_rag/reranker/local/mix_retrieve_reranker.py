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

import math
from typing import List, Union, Dict
from collections import defaultdict
from loguru import logger

from langchain_core.documents import Document

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import (validate_params, validate_list_document, MAX_QUERY_LENGTH,
                                 MAX_BATCH_SIZE, TEXT_MAX_LEN, STR_MAX_LEN, MAX_TOP_K, MB)


class MixRetrieveReranker(Reranker):

    @validate_params(
        k=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_TOP_K,
               message="param must be int and value range [1, 10000]"),
        baseline=dict(validator=lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
                      message="baseline must be a float in [0.0, 1.0]"),
        amplitude=dict(validator=lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
                       message="amplitude must be a float in [0.0, 1.0]"),
        slope=dict(validator=lambda x: isinstance(x, (int, float)) and x > 0,
                   message="slope must be a positive number (controls logistic curve steepness)"),
        midpoint=dict(validator=lambda x: isinstance(x, (int, float)) and x > 0,
                      message="midpoint must be a positive number"),
    )
    def __init__(self,
                 k: int = 100,
                 baseline: float = 0.4,
                 amplitude: float = 0.3,
                 slope: float = 1,
                 midpoint: float = 6):
        super(MixRetrieveReranker, self).__init__(k)
        self.k = k
        self.baseline = baseline
        self.amplitude = amplitude
        self.slope = slope
        self.midpoint = midpoint

    @staticmethod
    def _min_max_normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
            
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.0 for _ in scores]
        
        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
        return normalized

    @staticmethod
    def _normalize_retrieval_results(dense_docs: List[Document], sparse_docs: List[Document]):
        """
        对 dense 和 sparse 检索结果分别进行 Max-Min 归一化
        """
        # 1. 提取 dense 得分并归一化
        dense_scores = [doc.metadata.get("score", 0.0) for doc in dense_docs]
        dense_normalized = MixRetrieveReranker._min_max_normalize(dense_scores)
        
        # 2. 提取 sparse 得分并归一化
        sparse_scores = [doc.metadata.get("score", 0.0) for doc in sparse_docs]
        sparse_normalized = MixRetrieveReranker._min_max_normalize(sparse_scores)
        
        # 3. 更新文档 metadata
        for doc, norm_score in zip(dense_docs, dense_normalized):
            doc.metadata["norm_score"] = norm_score
        for doc, norm_score in zip(sparse_docs, sparse_normalized):
            doc.metadata["norm_score"] = norm_score

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MB,
                   message="param length range [1, 1024 * 1024]"),
        texts=dict(validator=lambda x: validate_list_document(x, [1, TEXT_MAX_LEN], [1, MB]),
                  message="param must meets: Type is List[docs], "
                          "list length range [1, 1000 * 1000], content length range [1, 1024 * 1024]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param value range [1, {MAX_BATCH_SIZE}]"),
    )
    def rerank(self,
               query: str,
               texts: list[Document],
               batch_size: int = 32) -> list[Document]:
        """
        对合并的 dense + sparse 检索结果进行归一化、加权融合、重排序
        """
        docs = texts

        dense_docs = [doc for doc in docs if doc.metadata.get('retrieval_type') == 'dense']
        sparse_docs = [doc for doc in docs if doc.metadata.get('retrieval_type') == 'sparse']

        # 归一化
        self._normalize_retrieval_results(dense_docs, sparse_docs)

        # 计算权重
        weight = self._logistic_func(len(query))

        # 按 doc_hash 聚合
        merged: Dict[str, Dict] = defaultdict(lambda: {
            "doc": None,
            "dense_score": 0.0,
            "sparse_score": 0.0,
        })

        for doc in dense_docs:
            doc_hash = hash(doc.page_content)
            merged[doc_hash]["doc"] = doc
            merged[doc_hash]["dense_score"] = doc.metadata["norm_score"]

        for doc in sparse_docs:
            doc_hash = hash(doc.page_content)
            if merged[doc_hash]["doc"] is not None:
                doc.metadata['retrieval_type'] = 'both'
            merged[doc_hash]["doc"] = doc
            merged[doc_hash]["sparse_score"] = doc.metadata["norm_score"]

        # 最终得分
        scored_docs = []
        for doc_hash, info in merged.items():
            final_score = weight * info['dense_score'] + (1.0 - weight) * info['sparse_score']
            scored_docs.append((final_score, info["doc"]))

        # 重排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Finished! Total Merged Docs: {len(merged)} | Dense: {len(dense_docs)} | Sparse: {len(sparse_docs)}")

        return [doc for _, doc in scored_docs[:self.k]]

    def _logistic_func(self, length: int):
        exponent = -self.slope * (length - self.midpoint)
        denominator = 1 + math.exp(exponent)
        return self.baseline + self.amplitude / denominator
