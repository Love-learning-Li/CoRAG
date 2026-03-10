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
from unittest.mock import MagicMock

import numpy as np

from mx_rag.graphrag.vector_stores.vector_store_wrapper import VectorStoreWrapper
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.vectorstore.vectorstore import VectorStore


class TestVectorStoreWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock(spec=VectorStore)
        self.wrapper = VectorStoreWrapper(self.mock_vector_store)

    def test_normalize_vectors_l2(self):
        vectors = np.array([[3.0, 4.0], [1.0, 2.0]]).astype(np.float32)
        VectorStoreWrapper.normalize_vectors_l2(vectors)
        expected_norms = np.array([1.0, 1.0], dtype=np.float32)
        actual_norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_almost_equal(actual_norms, expected_norms)

    def test_add(self):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
        ids = [1, 2]
        self.wrapper.add(vectors, ids)
        self.mock_vector_store.add.assert_called_once_with(ids, vectors)

    def test_search(self):
        query_vectors = np.array([[1.0, 0.0]])
        top_k = 1
        self.mock_vector_store.search.return_value = (np.array([0.1]), np.array([0]))
        distances, indices = self.wrapper.search(query_vectors, top_k)
        self.mock_vector_store.search.assert_called_once_with(query_vectors.tolist(), top_k)
        self.assertEqual(distances, [0.1])
        self.assertEqual(indices, [0])

    def test_ntotal_mindfaiss(self):
        self.wrapper.vector_store = MagicMock(spec=MindFAISS)
        self.wrapper.vector_store.get_ntotal.return_value = 10
        total = self.wrapper.ntotal()
        self.wrapper.vector_store.get_ntotal.assert_called_once()
        self.assertEqual(total, 10)

    def test_ntotal_other(self):
        self.mock_vector_store.get_all_ids.return_value = [1, 2, 3]
        total = self.wrapper.ntotal()
        self.mock_vector_store.get_all_ids.assert_called_once()
        self.assertEqual(total, 3)

    def test_clear(self):
        self.mock_vector_store.get_all_ids.return_value = [1, 2, 3]
        self.wrapper.clear()
        self.mock_vector_store.delete.assert_called_once_with([1, 2, 3])

    def test_save_mindfaiss(self):
        self.wrapper.vector_store = MagicMock(spec=MindFAISS)
        self.wrapper.save()
        self.wrapper.vector_store.save_local.assert_called_once()


if __name__ == "__main__":
    unittest.main()
