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

"""
reranker的工厂类，用于生产mxrag的reranker
"""
from abc import ABC
from typing import Dict, Any, Callable
from loguru import logger

from mx_rag.reranker.reranker import Reranker
from mx_rag.reranker.local import LocalReranker
from mx_rag.reranker.service import TEIReranker


class RerankerFactory(ABC):
    """
    功能描述:
        reranker的工厂方法类，用于生产mxrag的reranker

    Attributes:
        _NPU_SUPPORT_RERANKER 字典，用于映射reranker和对应的构造函数
    """
    _NPU_SUPPORT_RERANKER: Dict[str, Callable[[Dict[str, Any]], Reranker]] = {
        "local_reranker": LocalReranker.create,
        "tei_reranker": TEIReranker.create
    }

    @classmethod
    def create_reranker(cls, **kwargs):
        """
        功能描述:
            构造vector storage

        Args:
            kwargs: Dict[str, Any] 构造reranker的参数
        Return:
            similarity: Reranker 返回的reranker的实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "similarity_type" not in kwargs:
            logger.error("need similarity_config param. ")
            return None

        similarity_type = kwargs.pop("similarity_type")

        if not isinstance(similarity_type, str):
            logger.error("similarity_type should be str type. ")
            return None

        if similarity_type not in cls._NPU_SUPPORT_RERANKER:
            logger.error(f"similarity_type is not support. {similarity_type}")
            return None

        creator = cls._NPU_SUPPORT_RERANKER.get(similarity_type)

        try:
            similarity = creator(**kwargs)
        except KeyError:
            logger.error(f"create reranker key error")
            return None
        except Exception:
            logger.error(f"exception occurred while constructing reranker")
            return None

        return similarity
