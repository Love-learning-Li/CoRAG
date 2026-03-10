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


import io

import unittest
from unittest import mock
from unittest.mock import patch

from PIL import Image, ImageChops

from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam

MOCK_IMAGE = Image.new("RGB", (200, 200), color=(73, 109, 137))


class MockResponse:
    def __init__(self, headers, status):
        img_byte_arr = io.BytesIO()
        img_format = "png"
        MOCK_IMAGE.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)
        self.content = img_byte_arr.getvalue()
        self.headers = headers
        self.status = status

    def read(self, amt):
        return self.content


def compare_images(img1, img2):
    """
    比较两张 PIL 图像对象是否相同。
    """
    if img1.size != img2.size:
        return False

    diff = ImageChops.difference(img1, img2)
    if diff.getbbox() is None:
        return True
    else:
        return False


class TestMindieVision(unittest.TestCase):
    def test_img(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse({
            "Content-Type": "application/json",
            "Content-Length": 200
        }, 200))):
            sd_model = Text2ImgMultiModel(model_name="sd", url="http://test:8888", client_param=ClientParam(use_http=True))
            res = sd_model.text2img(prompt="dog wearing black glasses", output_format="png")
            self.assertNotEqual(res["result"], "")

    def test_img_interrupt(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse({
            "Content-Type": "application/json",
            "Content-Length": 200
        }, 404))):
            sd_model = Text2ImgMultiModel(model_name="sd", url="http://test:8888", client_param=ClientParam(use_http=True))
            res = sd_model.text2img(prompt="dog wearing black glasses", output_format="png")
            self.assertEqual(res["result"], "")


if __name__ == "__main__":
    unittest.main()
