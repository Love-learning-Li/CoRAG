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

from langchain_core.documents import Document

from mx_rag.chain import Img2ImgChain
from mx_rag.llm import Img2ImgMultiModel, LLMParameterConfig
from mx_rag.retrievers import Retriever


class TestImgChain(unittest.TestCase):
    def test_query(self):
        model = MagicMock(spec=Img2ImgMultiModel)
        retriever = MagicMock(spec=Retriever)
        chain = Img2ImgChain(model, retriever)
        llm_config = MagicMock(spec=LLMParameterConfig)
        # 检索文档为空
        retriever.invoke.return_value = [Document("")]
        result = chain.query("question", llm_config)
        self.assertEqual(result, {})
        # 检索文档不为空
        retriever.invoke.return_value = [Document("这是被切分的chunk")]
        model.img2img.return_value = "模型返回结果"
        result = chain.query("question", llm_config)
        self.assertEqual(result, "模型返回结果")
