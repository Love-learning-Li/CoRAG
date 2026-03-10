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
import os
import unittest
from unittest import mock
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
from urllib3 import HTTPResponse

from mx_rag.llm import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils import ClientParam


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


CONTENT = ('根据题目所给信息，我们可以知道这道题考查的是《赵氏孤儿》中的人物。'
           '因此，我们需要回忆起这部作品中的主要人物。'
           '\n\n经过查阅资料和记忆，我们得知，《赵氏孤儿》中的人物有程婴、公孙杵臼等。'
           '而选项A正是《赵氏孤儿》的别名之一，因此答案为A。')

RESPONSE = {
    'id': '99',
    'object': 'chat.completion',
    'created': 1716561049,
    'model': 'llama2-7b-hf',
    'choices': [{
        'index': 0,
        'message': {'role': 'assistant',
                    'content': CONTENT},
        'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 46, 'completion_tokens': 78, 'total_tokens': 124}
}

RESPONSE_STREAM = {
    'id': '99',
    'object': 'chat.completion',
    'created': 1716561049,
    'model': 'llama2-7b-hf',
    'choices': [{
        'index': 0,
        'delta': {'role': 'assistant',
                  'content': CONTENT},
        'finish_reason': None}],
    'usage': {'prompt_tokens': 46, 'completion_tokens': 78, 'total_tokens': 124}
}


class TestMindieLLM(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_chat(self):
        with patch("urllib3.PoolManager.request", mock.Mock(
                return_value=MockResponse(RESPONSE, {
                    "Content-Type": "application/json",
                    "Content-Length": 200
                }, 200))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            data = llm_model.chat(
                query="程婴、公孙杵臼是____中的人物。\nA. 《赵氏孤儿》\nB. 《杀狗记》\nC. 《墙头马上》\nD. 《岳阳楼》",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            self.assertEqual(data, CONTENT)

    def test_chat_finish_reason_length(self):
        RESPONSE['choices'][0]['finish_reason'] = "length"
        with patch("urllib3.PoolManager.request", mock.Mock(
                return_value=MockResponse(RESPONSE, {
                    "Content-Type": "application/json",
                    "Content-Length": 200
                }, 200))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            data = llm_model.chat(
                query="程婴、公孙杵臼是____中的人物。\nA. 《赵氏孤儿》\nB. 《杀狗记》\nC. 《墙头马上》\nD. 《岳阳楼》",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            self.assertEqual(data, CONTENT + "......")
        RESPONSE['choices'][0]['finish_reason'] = "stop"

    def test_chat_json_error(self):
        with patch("urllib3.PoolManager.request", mock.Mock(
                return_value=MockResponse(json_data="invalid json", headers={
                    "Content-Type": "application/json",
                    "Content-Length": 200
                }, status=200))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            data = llm_model.chat(
                query="程婴、公孙杵臼是____中的人物。\nA. 《赵氏孤儿》\nB. 《杀狗记》\nC. 《墙头马上》\nD. 《岳阳楼》",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            self.assertEqual(data, "")

    def test_chat_stream(self):
        # 模拟成功的流式响应
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/event-stream"}

        # 模拟流式响应的数据块
        data_blocks = [
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n",
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n",
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n"
        ]

        # 模拟stream方法，逐步返回数据块
        def mock_stream(chunk_size):
            for block in data_blocks:
                yield block

        mock_response.stream = mock_stream

        with patch("urllib3.PoolManager.request", return_value=mock_response):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            stream_data = llm_model.chat_streamly(
                query="test query",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            expected_output = ["Hello", "Hello World"]
            for i, data in enumerate(stream_data):
                self.assertEqual(data, expected_output[i])

    def test_chat_stream_length(self):
        # 模拟失败的流式响应
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/event-stream"}

        # 模拟流式响应的数据块
        data_blocks = [
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n",
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n",
            b"data: {\"id\":\"99\",\"object\":\"chat.completion\",\"created\":1716561049,\"model\":\"llama2-7b-hf\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":46,\"completion_tokens\":78,\"total_tokens\":124}}\n"
        ]

        # 模拟stream方法，逐步返回数据块
        def mock_stream(chunk_size):
            for block in data_blocks:
                yield block

        mock_response.stream = mock_stream

        with patch("urllib3.PoolManager.request", return_value=mock_response):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            stream_data = llm_model.chat_streamly(
                query="test query"
            )
            expected_output = ["Hello", "Hello World", "Hello World......"]
            for i, data in enumerate(stream_data):
                self.assertEqual(data, expected_output[i])

    def test_chat_stream_json_error(self):
        # 模拟失败的流式响应
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/event-stream"}

        # 模拟流式响应的数据块
        data_blocks = [
            b"data: invalid json",
        ]

        # 模拟stream方法，逐步返回数据块
        def mock_stream(chunk_size):
            for block in data_blocks:
                yield block

        mock_response.stream = mock_stream

        with patch("urllib3.PoolManager.request", return_value=mock_response):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            stream_data = llm_model.chat_streamly(
                query="test query",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            expected_output = ["Hello", "Hello World"]
            for i, data in enumerate(stream_data):
                self.assertEqual(data, expected_output[i])

    def test_chat_interrupt(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse(RESPONSE_STREAM, {
            "Content-Type": "application/json",
            "Content-Length": 200
        }, 404))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            data = llm_model.chat(query="你好", sys_messages=[], llm_config=LLMParameterConfig(max_tokens=1024))
            self.assertEqual(data, "")

    def test_chat_stream_interrupt(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse(RESPONSE_STREAM, {
            "Content-Type": "text/event-stream",
        }, 404))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                                     client_param=ClientParam(use_http=True))
            stream_data = llm_model.chat_streamly(query="你好", sys_messages=[],
                                                  llm_config=LLMParameterConfig(max_tokens=1024))
            data = False
            for _ in stream_data:
                data = True
            self.assertFalse(data)

    def test_chat_param_max_tokens(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(max_tokens=0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_history(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        sys_messages = [{"role": "users", "content": "test"}] * 101
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_history_2(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        sys_messages = [{"role": "users", "content": "test", "111": "1"}]
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = True
        self.assertFalse(error)

    def test_chat_param_history_3(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        sys_messages = [{"role": "user", "contentcontentcontentcontent": "test", "111": "1"}]
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = False
        self.assertFalse(error)

    def test_chat_param_presence_penalty(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(presence_penalty=-5.0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_presence_penalty_2(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                                 client_param=ClientParam(use_http=True))
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(presence_penalty=-5.0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_with_cert(self):
        cart_file = os.path.join(self.current_dir, "../../data/root_ca.crt")
        cart_file = os.path.realpath(cart_file)
        try:
            Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888",
                         client_param=ClientParam(ca_file=cart_file))
        except Exception as e:
            logger.info(e)


if __name__ == '__main__':
    unittest.main()
