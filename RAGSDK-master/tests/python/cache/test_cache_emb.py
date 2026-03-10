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
from unittest import mock
from unittest.mock import patch

from mx_rag.cache.cache_emb.cache_emb import CacheEmb
from mx_rag.embedding import EmbeddingFactory


class TestCacheEmb(unittest.TestCase):
    def test_cache_emb_init_type_exception(self):
        emb = EmbeddingFactory.create_embedding(**{
            "x_dim": 1024,
            "skip_emb": False,
            "embedding_type": "xxxx"  # error happen,
        })
        self.assertEqual(emb, None)

        emb = EmbeddingFactory.create_embedding(**{
            "x_dim": 1024,
            "skip_emb": False,  # no embedding_type error
        })
        self.assertEqual(emb, None)

        emb = EmbeddingFactory.create_embedding(**{
            "x_dim": 1024,
            "skip_emb": False,
            "embedding_type": 1234  # type error
        })
        self.assertEqual(emb, None)

    def test_cache_emb(self):
        def mock_create_embedding(*args, **kwargs):
            return None

        with patch('mx_rag.embedding.embedding_factory.EmbeddingFactory.create_embedding',
                   mock.Mock(side_effect=mock_create_embedding)):
            cache_emb = CacheEmb.create(embedding_type="xxxx", x_dim=1024, skip_emb=False)
            self.assertIsInstance(cache_emb, CacheEmb)
            self.assertEqual(cache_emb.dimension(), 1024)
            try:
                cache_emb.to_embeddings(["1234567"])
            except Exception as e:
                self.assertEqual(f"{e}", "emb_obj is not instance of Embeddings")


if __name__ == '__main__':
    unittest.main()
