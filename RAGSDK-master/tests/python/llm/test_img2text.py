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
from unittest.mock import patch
import json

from mx_rag.llm import Img2TextLLM, LLMParameterConfig
from mx_rag.llm.img2text import _check_image_url
from mx_rag.utils import ClientParam

# 定义一些常量（假设）
MAX_URL_LENGTH = 1024
MAX_PROMPT_LENGTH = 1024
MAX_MODEL_NAME_LENGTH = 1024
MB = 1024 * 1024


RESPONSE = {
    'id': '99',
    'object': 'chat.completion',
    'created': 1716561049,
    'model': 'llama2-7b-hf',
    'choices': [{
        'index': 0,
        'message': {'role': 'assistant',
                    'content': "测试图像描述"},
        'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 46, 'completion_tokens': 78, 'total_tokens': 124}
}

class MockResponse:
    def __init__(self, json_data, headers, status):
        self.json_data = json_data
        self.headers = headers
        self.status = status

    def stream(self, chunk_size):
        for content in self.json_data["choices"][0]["delta"]["content"]:
            mock_response = self.json_data.copy()
            mock_response["choices"][0]["delta"]["content"] = content
            yield bytes('data: ' + json.dumps(mock_response) + "\n", 'utf-8')
        mock_response = self.json_data.copy()
        mock_response["choices"][0]["delta"] = {}
        mock_response["choices"][0]["finish_reason"] = "stop"
        yield bytes('data: ' + json.dumps(mock_response), 'utf-8')

    def read(self, amt):
        if isinstance(self.json_data, str):
            return self.json_data
        return bytes(json.dumps(self.json_data), 'utf-8')

class TestImg2TextLLM(unittest.TestCase):
    def setUp(self):
        # 初始化测试环境
        self.base_url = "http://example.com"
        self.model_name = "test_model"
        self.llm_config = LLMParameterConfig()
        self.client_param = ClientParam(timeout=5, use_http=True)

        # 创建 Img2TextLLM 实例
        self.img2text = Img2TextLLM(
            base_url=self.base_url,
            model_name=self.model_name,
            llm_config=self.llm_config,
            client_param=self.client_param
        )

    @patch("urllib3.PoolManager.request")
    def test_chat_success(self, mock_request_utils):
        # 模拟成功的 HTTP 请求
        RESPONSE['choices'][0]['finish_reason'] = "stop"
        mock_request_utils.return_value = MockResponse(RESPONSE, {
                    "Content-Type": "application/json",
                    "Content-Length": 200
                }, 200)

        # 测试输入
        image_url = {"url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}

        # 调用 chat 方法
        result = self.img2text.chat(image_url=image_url)

        # 验证结果
        self.assertEqual(result, "测试图像描述")


    @patch("urllib3.PoolManager.request")
    def test_chat_failure(self, mock_request_utils):
        # 模拟失败的 HTTP 请求
        mock_request_utils.return_value = MockResponse(RESPONSE, {
            "Content-Type": "application/json",
            "Content-Length": 200
        }, 400)

        # 测试输入
        image_url = {"url": "data:image/jpeg;base64,iVBORw0K"}

        # 调用 chat 方法
        result = self.img2text.chat(image_url=image_url)

        # 验证结果
        self.assertEqual(result, "")

        with self.assertRaises(ValueError):
            image_url = ["data:image/jpeg;base64,base64_string"]
            self.img2text.chat(image_url=image_url)

        with self.assertRaises(ValueError):
            image_url = {"url": ""}
            self.img2text.chat(image_url=image_url)
        with self.assertRaises(ValueError):
            img2text = Img2TextLLM(
                base_url=self.base_url,
                model_name=self.model_name,
                llm_config=self.llm_config,
                client_param=self.client_param,
                prompt=""
            )
            image_url = {"url": "data:image/jpeg;base64,base64_string"}
            img2text.chat(image_url=image_url)


    def test_check_image_url_valid(self):
        # 测试有效的 image_url
        image_url = {"url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}
        self.assertTrue(_check_image_url(image_url))

    def test_check_image_url_invalid(self):
        # 测试无效的 image_url
        invalid_urls = [
            {},  # 空字典
            {"url": 123},  # url 不是字符串
            {"key": "value"}  # 不包含 url 键
        ]
        for url in invalid_urls:
            self.assertFalse(_check_image_url(url))


if __name__ == "__main__":
    unittest.main()