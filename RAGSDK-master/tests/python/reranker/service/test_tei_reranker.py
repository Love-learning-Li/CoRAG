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


import json
import random
from typing import Dict
import unittest
from unittest import mock
from unittest.mock import patch

from mx_rag.reranker.service import TEIReranker
from mx_rag.utils import ClientParam


class TestTEIReranker(unittest.TestCase):
    class Result:
        def __init__(self, success: bool, data: str):
            self.success = success
            self.data = data

    def test_request_success(self):
        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['texts'])):
                response_data.append({'index': i, 'score': random.random()})
            return TestTEIReranker.Result(True, json.dumps(response_data))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            rerank = TEIReranker(url='https://localhost:8888/rerank', client_param=ClientParam(use_http=True))

            texts = ['我是小黑', '我是小红'] * 100
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

            texts = ['我是小黑', '我是小红'] * 300
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

    def test_request_success_1(self):
        def mock_post(url: str, body: str, headers: Dict):
            data = json.loads(body)
            response_data = []
            for i in range(len(data['documents'])):
                response_data.append({'index': i, 'relevance_score': random.random()})
            return TestTEIReranker.Result(True, json.dumps({"results": response_data}))

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            rerank = TEIReranker(url='https://localhost:8888/v1/rerank', client_param=ClientParam(use_http=True))

            texts = ['我是小黑', '我是小红'] * 100
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

            texts = ['我是小黑', '我是小红'] * 300
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (len(texts),))

    def test_empty_texts(self):
        rerank = TEIReranker(url='https://localhost:8888/rerank', client_param=ClientParam(use_http=True))

        texts = ["text"]
        scores = rerank.rerank(query='你好', texts=texts)
        self.assertEqual(scores.shape, (0,))

    def test_texts_too_long(self):
        rerank = TEIReranker(url='https://localhost:8888/rerank', client_param=ClientParam(use_http=True))

        texts = ['我是小黑', '我是小红'] * 500001
        with self.assertRaises(ValueError):
            rerank.rerank(query='你好', texts=texts)

    def test_request_failed(self):
        def mock_post(url: str, body: str, headers: Dict):
            return TestTEIReranker.Result(False, "")

        with patch('mx_rag.utils.url.RequestUtils.post', mock.Mock(side_effect=mock_post)):
            rerank = TEIReranker(url='https://localhost:8888/rerank', client_param=ClientParam(use_http=True))

            texts = ['我是小黑', '我是小红'] * 300
            scores = rerank.rerank(query='你好', texts=texts)
            self.assertEqual(scores.shape, (0,))


if __name__ == '__main__':
    unittest.main()
