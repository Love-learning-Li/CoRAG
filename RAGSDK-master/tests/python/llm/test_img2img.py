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
from unittest.mock import patch, Mock
from mx_rag.llm import Img2ImgMultiModel


class TestImg2ImgMultiModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch constants and decorators before importing the class
        cls.patcher_validate_params = patch('mx_rag.utils.common.validate_params', lambda **kwargs: (lambda f: f))
        cls.patcher_validate_params.start()
        cls.patcher_client_param = patch('mx_rag.utils.ClientParam', autospec=True)
        cls.MockClientParam = cls.patcher_client_param.start()
        cls.patcher_max_url_length = patch('mx_rag.utils.common.MAX_URL_LENGTH', 128)
        cls.patcher_max_url_length.start()
        cls.patcher_max_model_name_length = patch('mx_rag.utils.common.MAX_MODEL_NAME_LENGTH', 128)
        cls.patcher_max_model_name_length.start()
        cls.patcher_mb = patch('mx_rag.utils.common.MB', 1024 * 1024)
        cls.patcher_mb.start()
        cls.patcher_max_prompt_length = patch('mx_rag.utils.common.MAX_PROMPT_LENGTH', 1024 * 1024)
        cls.patcher_max_prompt_length.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher_validate_params.stop()
        cls.patcher_client_param.stop()
        cls.patcher_max_url_length.stop()
        cls.patcher_max_model_name_length.stop()
        cls.patcher_mb.stop()
        cls.patcher_max_prompt_length.stop()

    def setUp(self):
        # Patch RequestUtils and logger at usage site for each test
        self.patcher_request_utils = patch('mx_rag.llm.img2img.RequestUtils', autospec=True)
        self.MockRequestUtils = self.patcher_request_utils.start()
        self.patcher_logger = patch('mx_rag.llm.img2img.logger')
        self.mock_logger = self.patcher_logger.start()

        self.url = "http://localhost"
        self.model_name = "test-model"
        self.client_param = self.MockClientParam()
        self.mock_client = Mock()
        self.MockRequestUtils.return_value = self.mock_client

    def tearDown(self):
        self.patcher_request_utils.stop()
        self.patcher_logger.stop()

    def test_init_sets_attributes(self):
        model = Img2ImgMultiModel(self.url, self.model_name, self.client_param)
        self.assertEqual(model._url, self.url)
        self.assertEqual(model._model_name, self.model_name)
        self.assertEqual(model._client, self.mock_client)
        self.assertEqual(model.headers, {'Content-Type': 'application/json'})

    def test_img2img_success(self):
        model = Img2ImgMultiModel(self.url, self.model_name, self.client_param)
        prompt = "draw a cat"
        image_content = "fake_image_data"
        size = "512*512"
        expected_image = "result_image_data"
        mock_response = Mock(success=True, data='{"image": "result_image_data"}')
        self.mock_client.post.return_value = mock_response

        result = model.img2img(prompt, image_content, size)
        self.assertEqual(result, {"prompt": prompt, "result": expected_image})

    def test_img2img_unsuccessful_response(self):
        model = Img2ImgMultiModel(self.url, self.model_name, self.client_param)
        prompt = "draw a dog"
        image_content = "img"
        size = "512*512"
        mock_response = Mock(success=False, data='{}')
        self.mock_client.post.return_value = mock_response

        result = model.img2img(prompt, image_content, size)
        self.assertEqual(result, {"prompt": prompt, "result": ""})
        self.mock_logger.error.assert_called()

    def test_img2img_json_decode_error(self):
        model = Img2ImgMultiModel(self.url, self.model_name, self.client_param)
        prompt = "draw a horse"
        image_content = "img"
        size = "512*512"
        mock_response = Mock(success=True, data='not a json')
        self.mock_client.post.return_value = mock_response

        result = model.img2img(prompt, image_content, size)
        self.assertEqual(result, {"prompt": prompt, "result": ""})
        self.mock_logger.error.assert_called()

    def test_img2img_missing_image_key(self):
        model = Img2ImgMultiModel(self.url, self.model_name, self.client_param)
        prompt = "draw a bird"
        image_content = "img"
        size = "512*512"
        mock_response = Mock(success=True, data='{"not_image": "nope"}')
        self.mock_client.post.return_value = mock_response

        result = model.img2img(prompt, image_content, size)
        self.assertEqual(result, {"prompt": prompt, "result": ""})
        self.mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
