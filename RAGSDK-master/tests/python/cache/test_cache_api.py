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

import os
import unittest
from unittest import mock
from unittest.mock import patch

from gptcache import Cache

from mx_rag.cache.cache_api.cache_init import _init_mxrag_cache
from mx_rag.cache import CacheConfig, SimilarityCacheConfig


class TestCacheApi(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_init_memory_api(self):
        data_save_folder = os.path.join(TestCacheApi.current_dir, "cache_save_folder")
        if not os.path.exists(data_save_folder):
            os.mkdir(data_save_folder)
        cache_config: CacheConfig = CacheConfig(
            cache_size=100,
            data_save_folder=data_save_folder
        )

        cache = Cache()
        mock_init_similar_cache = mock.Mock(return_value=None)
        with patch('gptcache.adapter.api.init_similar_cache', mock_init_similar_cache):
            _init_mxrag_cache(cache, "test_init_memory_config", cache_config)
        mock_init_similar_cache.assert_called_once()

    @patch("mx_rag.cache.cache_storage.cache_vec_storage.CacheVecStorage.create")
    @patch("mx_rag.cache.cache_similarity.cache_similarity.CacheSimilarity.create")
    @patch("mx_rag.cache.cache_emb.cache_emb.CacheEmb.create")
    @patch("gptcache.manager.get_data_manager")
    def test_init_similarity_api(self, vector_create, similarity_create, emb_create, mock_get_data_manager):
        vector_create.return_value = None
        similarity_create.return_value = None
        emb_create.return_value = None
        mock_get_data_manager.return_value = None

        dim = 0
        npu_dev = 0
        data_save_folder = os.path.join(TestCacheApi.current_dir, "cache_save_folder")
        cache_config: SimilarityCacheConfig = SimilarityCacheConfig(
            vector_config={
                "vector_type": "npu_faiss_db",
                "x_dim": dim,
                "devs": [npu_dev],
            },
            cache_config="sqlite",
            emb_config={
                "embedding_type": "local_text_embedding",
                "x_dim": dim,
                "model_path": "/data/acge_text_embedding/",
                "dev_id": npu_dev
            },
            similarity_config={
                "similarity_type": "local_reranker",
                "model_path": "/data/bge-reranker-v2-m3",
                "dev_id": npu_dev
            },
            retrieval_top_k=5,
            cache_size=1000,
            similarity_threshold=0.8,
            data_save_folder=data_save_folder
        )

        cache = Cache()
        mock_init_similar_cache = mock.Mock(return_value=None)
        with patch('gptcache.adapter.api.init_similar_cache', mock_init_similar_cache):
            _init_mxrag_cache(cache, "test_init_similarity_config", cache_config)

        mock_init_similar_cache.assert_called_once()


if __name__ == '__main__':
    unittest.main()
