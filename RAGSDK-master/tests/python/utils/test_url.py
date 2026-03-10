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
from unittest.mock import patch, MagicMock

from urllib3.exceptions import TimeoutError as urllib3_TimeoutError, HTTPError
from mx_rag.utils import ClientParam
from mx_rag.utils.url import is_url_valid, RequestUtils


class TestURL(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cart_file = os.path.realpath(os.path.join(current_dir, "../../data/root_ca.crt"))

    def test_is_url_valid(self):
        self.assertFalse(is_url_valid("https://www.google.com", True))
        self.assertTrue(is_url_valid("https://www.google.com", False))
        self.assertTrue(is_url_valid("http://www.google.com", True))
        self.assertFalse(is_url_valid("http://www.google.com", False))
        self.assertFalse(is_url_valid("not a url", True))

    def test_check_ca_content(self):
        with self.assertRaises(ValueError):
            client_param = ClientParam(use_http=False, ca_file=self.cart_file)
            request_utils = RequestUtils(client_param=client_param)
            request_utils._check_ca_content(self.cart_file)

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = FileNotFoundError
            with self.assertRaises(ValueError):
                client_param = ClientParam(use_http=False, ca_file=self.cart_file)
                request_utils = RequestUtils(client_param=client_param)
                request_utils._check_ca_content(self.cart_file)

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError
            with self.assertRaises(ValueError):
                client_param = ClientParam(use_http=False, ca_file=self.cart_file)
                request_utils = RequestUtils(client_param=client_param)
                request_utils._check_ca_content(self.cart_file)

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("Unknown error")
            with self.assertRaises(ValueError):
                client_param = ClientParam(use_http=False, ca_file=self.cart_file)
                request_utils = RequestUtils(client_param=client_param)
                request_utils._check_ca_content(self.cart_file)

    def test_post(self):
        with patch('mx_rag.utils.url.is_url_valid') as mock_is_url_valid:
            mock_is_url_valid.return_value = False
            client_param = ClientParam(use_http=True, timeout=10)
            request_utils = RequestUtils(client_param=client_param)
            result = request_utils.post('http://test.com', 'body', {'headers': 'headers'})
            self.assertFalse(result.success)
            result = request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'})
            self.assertFalse(list(result)[0].success)

        with patch('mx_rag.utils.url.is_url_valid') as mock_is_url_valid:
            mock_is_url_valid.return_value = True
            with patch("mx_rag.utils.url.urllib3.PoolManager") as mock_pool_manager:

                mock_pool = mock_pool_manager.return_value
                mock_pool.request.side_effect = urllib3_TimeoutError
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)
                result = request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(list(result)[0].success)

                mock_pool.request.side_effect = HTTPError
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)
                result = request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(list(result)[0].success)

                mock_pool.request.side_effect = Exception
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)
                result = request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'})
                self.assertFalse(list(result)[0].success)

            with patch("mx_rag.utils.url.urllib3.PoolManager") as mock_pool_manager:
                mock_pool = mock_pool_manager.return_value
                # 模拟成功响应对象
                mock_response = MagicMock(headers={'Content-Length': '10', 'Content-Type': 'text/event-stream'}, status=200)  # 模拟有效的 Content-Length
                mock_response.read.return_value = b"response data"
                mock_pool.request.return_value = mock_response
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://www.google.com', 'body', {'headers': 'headers'})
                self.assertTrue(result.success)
                result = list(request_utils.post_streamly('http://www.google.com', 'body', {'headers': 'headers'}))
                self.assertEqual(result, [])

                mock_response = MagicMock(headers={'Content-Length': 'invalid', 'Content-Type': None}, status=200)
                mock_response.read.return_value = b"response data"
                mock_pool.request.return_value = mock_response
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://www.google.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)
                result = list(request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'}))
                self.assertFalse(result[0].success)

                mock_response = MagicMock(headers={'Content-Length': '5000', 'Content-Type': 'test'}, status=200)
                mock_response.read.return_value = b"response data"
                mock_pool.request.return_value = mock_response
                request_utils = RequestUtils(client_param=client_param)
                request_utils.response_limit_size = 1000
                result = request_utils.post('http://www.google.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)
                result = list(request_utils.post_streamly('http://test.com', 'body', {'headers': 'headers'}))
                self.assertFalse(result[0].success)

                mock_response = MagicMock(headers={'Content-Length': '10', 'Content-Type': 'text/event-stream'}, status=400)
                mock_response.read.return_value = b"response data"
                mock_pool.request.return_value = mock_response
                request_utils = RequestUtils(client_param=client_param)
                result = request_utils.post('http://www.google.com', 'body', {'headers': 'headers'})
                self.assertFalse(result.success)



