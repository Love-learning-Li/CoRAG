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

from typing import Dict

from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params, TEXT_MAX_LEN
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain.base import Chain
from mx_rag.llm import Img2ImgMultiModel


class Img2ImgChain(Chain):
    """ 检索出输入文本最相关的图片与prompt合并发送给大模型，生成相应图片 """

    @validate_params(
        multi_model=dict(validator=lambda x: isinstance(x, Img2ImgMultiModel),
                         message="param must be instance of Img2ImgMultiModel"),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever),
                       message="param must be instance of BaseRetriever")
    )
    def __init__(self, multi_model, retriever):
        self._multi_model = multi_model
        self._retriever = retriever

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                  message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) -> Dict:
        image_content = self._retrieve_img(text)
        if not image_content:
            logger.error("retrieve similarity image failed")
            return {}

        return self._multi_model.img2img(prompt=kwargs.get("prompt"), image_content=image_content,
                                         size=kwargs.get("size", "512*512"))

    def _retrieve_img(self, text: str) -> str:
        """ 从向量数据库中检视text最相近的图片 """
        docs = self._retriever.invoke(text)
        if not isinstance(docs, list) or len(docs) == 0:
            logger.error("retrieve similarity image failed")
            return ""

        return docs[0].page_content
