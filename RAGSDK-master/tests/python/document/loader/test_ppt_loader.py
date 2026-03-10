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

from mx_rag.document.loader.ppt_loader import PowerPointLoader


class TestPPTLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_load(self):
        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"))
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=True)
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

    def test_enable_ocr_false(self):
        # 禁用ocr进行图片识别
        loader = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=False)
        ppt_doc = loader.load()
        self.assertEqual(ppt_doc[0].metadata["source"],
                         os.path.join(self.data_dir, "test.pptx"))

    def test_invalid_enable_ocr(self):
        with self.assertRaises(ValueError):
            _ = PowerPointLoader(os.path.join(self.data_dir, "test.pptx"), enable_ocr=0)
