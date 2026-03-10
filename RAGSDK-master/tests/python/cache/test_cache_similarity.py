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

from mx_rag.reranker import RerankerFactory
from mx_rag.reranker.reranker import Reranker
from mx_rag.cache.cache_similarity.cache_similarity import CacheSimilarity
from cache_mocker import MockerReranker


def mock_create_similarity(*args, **kwargs):
    return MockerReranker(0)


class TestCacheSimilarity(unittest.TestCase):
    def test_cache_similarity_init_exception(self):
        reranker = RerankerFactory.create_reranker(**{
            "similarity_type": 1234  # type error
        })
        self.assertEqual(reranker, None)

        reranker = RerankerFactory.create_reranker(**{
            "similarity_type": "xxxx"  # value error
        })
        self.assertEqual(reranker, None)

        reranker = RerankerFactory.create_reranker(**{
            "xxxx": "xxxx"  # type error
        })
        self.assertEqual(reranker, None)

    def test_cache_similarity(self):
        with patch('mx_rag.reranker.reranker_factory.RerankerFactory.create_reranker',
                   mock.Mock(side_effect=mock_create_similarity)):
            similarity = CacheSimilarity.create(similarity=1234)

            src_dict = {
                "question": "hello world",
            }

            cache_dict = {
                "question": "hello world",
            }
            self.assertEqual(similarity.evaluation(src_dict, cache_dict), 1)

            cache_dict = {
                "question": "hello",
            }

            self.assertEqual(similarity.evaluation(src_dict, cache_dict), 0)

    def test_cache_similarity_range(self):
        with patch('mx_rag.reranker.reranker_factory.RerankerFactory.create_reranker',
                   mock.Mock(side_effect=mock_create_similarity)):
            similarity = CacheSimilarity.create(similarity_type=1234)

            score_min, score_max = similarity.range()
            self.assertEqual(score_min, 0.0)
            self.assertEqual(score_max, 1.0)

            similarity = CacheSimilarity.create(similarity_type=1234, score_min=2.0, score_max=4.0)

            score_min, score_max = similarity.range()
            self.assertEqual(score_min, 2.0)
            self.assertEqual(score_max, 4.0)


if __name__ == '__main__':
    unittest.main()
