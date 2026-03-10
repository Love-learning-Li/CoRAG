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

from mx_rag.cache import CacheChainChat, MxRAGCache
from mx_rag.chain import SingleText2TextChain
from mx_rag.llm import LLMParameterConfig


def _convert_data_to_user(data):
    return data

def _convert_data_to_cache(data):
    return data


class TestCacheChain(unittest.TestCase):
    def test_query(self):
        cache = MagicMock(spec=MxRAGCache)
        text2text_chain = MagicMock(spec=SingleText2TextChain)
        cache_chain = CacheChainChat(cache, text2text_chain, _convert_data_to_cache, _convert_data_to_user)
        llm_config = MagicMock(spec=LLMParameterConfig)
        # json无法解析时直接返回原值
        cache.search.return_value = "return value test"
        result = cache_chain.query("test", llm_config)
        self.assertEqual(result, "return value test")
        # 能解析json调用_convert_data_to_user处理返回值
        res = {"query": "\u9ad8\u8003\u9898\u76ee\u662f\u4ec0\u4e48\uff1f", "result": "\u8bed\u6587\u5168\u56fd\u5377"}
        cache.search.return_value = res
        result = cache_chain.query("test", llm_config)

        self.assertEqual(result, res)
        # search为None的情况
        cache.search.return_value = None
        text2text_chain.query.return_value = {"query": "test", "result": "大模型返回"}

        cache.update.return_value = None

        result = cache_chain.query("test", llm_config)

        self.assertEqual(result, {'query': 'test', 'result': '大模型返回'})


if __name__ == '__main__':
    unittest.main()
