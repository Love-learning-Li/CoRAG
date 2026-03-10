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

from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain.base import Chain
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils.common import validate_params, MAX_PROMPT_LENGTH


class Text2ImgChain(Chain):
    """使用 Text2ImgMultiModel 从文本提示生成图像的链。"""

    @validate_params(
        multi_model=dict(validator=lambda x: isinstance(x, Text2ImgMultiModel),
                         message="param must be instance of Text2ImgMultiModel")
    )
    def __init__(self, multi_model):
        """初始化 Text2ImgChain。

        参数:
            multi_model: 用于生成图像的 Text2ImgMultiModel。
        """
        self._multi_model = multi_model

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                  message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) -> Dict:
        """从给定的文本提示生成图像。

        参数:
            text: 用于生成图像的文本提示。
            llm_config: 用于查询的 LLM 配置 (此实现中未使用)。
            *args: 其他位置参数 (此实现中未使用)。
            **kwargs: 可以传递给 Text2ImgMultiModel 的其他关键字参数。

        返回:
            包含生成图像的字典。
        """
        return self._multi_model.text2img(prompt=text,
                                          output_format=kwargs.get("output_format", "png"),
                                          size=kwargs.get("size", "512*512"))
