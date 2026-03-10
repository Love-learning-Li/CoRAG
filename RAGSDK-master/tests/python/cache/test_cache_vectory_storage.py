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

from typing import List
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
from gptcache.manager.vector_data.base import VectorData

from mx_rag.cache.cache_storage.cache_vec_storage import CacheVecStorage
from mx_rag.storage.vectorstore.vector_storage_factory import VectorStorageError
from cache_mocker import MockerVecStorage


def mock_create_vector_storage(*args, **kwargs):
    return MockerVecStorage()


class TestCacheVectorStorage(unittest.TestCase):
    def test_cache_vector_storage_init_exception(self):
        self.assertRaises(VectorStorageError, CacheVecStorage.create, **{
            "error key": 69,
        })

        self.assertRaises(VectorStorageError, CacheVecStorage.create, **{
            "vector_type": 1234,
        })

        self.assertRaises(VectorStorageError, CacheVecStorage.create, **{
            "vector_type": "hello world",
        })

    def test_cache_vector_storage_add_delete(self):
        with patch('mx_rag.storage.vectorstore.vector_storage_factory.VectorStorageFactory.create_storage',
                   mock.Mock(side_effect=mock_create_vector_storage)):
            cache_vector_storage = CacheVecStorage.create(
                vector_type="mockVectorStorage",
                top_k=10,
                vector_save_file="./local_file.index"
            )

            add_data: List[VectorData] = [
                VectorData(0, np.zeros(shape=(1, 3))),
                VectorData(1, np.zeros(shape=(1, 3))),
                VectorData(2, np.zeros(shape=(1, 3)))
            ]

            self.assertEqual(cache_vector_storage.count(), 10)

            vector_add_mock = mock.Mock(return_value=None)
            with patch('cache_mocker.MockerVecStorage.add', vector_add_mock):
                cache_vector_storage.mul_add(add_data)

            vector_add_mock.assert_called_once()

            vector_delete_mock = mock.Mock(return_value=None)
            with patch('cache_mocker.MockerVecStorage.delete', vector_delete_mock):
                cache_vector_storage.delete(0)

            vector_delete_mock.assert_called_once()
            vector_delete_mock.assert_called_with(0)

    def test_cache_vector_storage_search(self):
        with patch('mx_rag.storage.vectorstore.vector_storage_factory.VectorStorageFactory.create_storage',
                   mock.Mock(side_effect=mock_create_vector_storage)):
            cache_vector_storage = CacheVecStorage.create(
                vector_type="mockVectorStorage",
                top_k=10,
                vector_save_file="./local_file.index"
            )

            vector_search_mock = mock.Mock(return_value=([[1]], [[1]]))
            with patch('cache_mocker.MockerVecStorage.search', vector_search_mock):
                result = cache_vector_storage.search(np.zeros(shape=(1, 3)), -1)
                self.assertEqual(result[0], (1, 1))

            vector_search_mock.assert_called_once()

    def test_cache_vector_storage_flush(self):
        with patch('mx_rag.storage.vectorstore.vector_storage_factory.VectorStorageFactory.create_storage',
                   mock.Mock(side_effect=mock_create_vector_storage)):
            cache_vector_storage = CacheVecStorage.create(
                vector_type="mockVectorStorage",
                top_k=10,
                vector_save_file="./local_file.index"
            )

            vector_flush_mock = mock.Mock(return_value=None)
            vector_save_file_mock = mock.Mock(return_value="user_file.txt")
            with patch('cache_mocker.MockerVecStorage.get_save_file',
                       vector_save_file_mock):
                with patch('cache_mocker.MockerVecStorage.save_local',
                           vector_flush_mock):
                    cache_vector_storage.flush()

            vector_save_file_mock.assert_called_once()
            vector_flush_mock.assert_called_once()


if __name__ == '__main__':
    unittest.main()
