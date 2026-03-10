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
embedding的工厂类，用于生产mxrag的embedding
"""
from abc import ABC
from typing import Dict, Any, Callable
from loguru import logger
from langchain_core.embeddings import Embeddings

from mx_rag.embedding.local import TextEmbedding, ImageEmbedding
from mx_rag.embedding.service import TEIEmbedding


class EmbeddingFactory(ABC):
    """
    功能描述:
        embedding的工厂方法类，用于生产mxrag的embedding

    Attributes:
        NPU_SUPPORT_EMB 字典，用于映射embedding和对应的构造函数
    """
    NPU_SUPPORT_EMB: Dict[str, Callable[[Dict[str, Any]], Embeddings]] = {
        "local_text_embedding": TextEmbedding.create,
        "local_images_embedding": ImageEmbedding.create,
        "tei_embedding": TEIEmbedding.create
    }

    @classmethod
    def create_embedding(cls, **kwargs) -> Embeddings:
        """
        功能描述:
            构造embedding

        Args:
            kwargs: Dict[str, Any] 构造embedding的参数
        Return:
            embedding: Embedding 返回的embedding的实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "embedding_type" not in kwargs:
            logger.error("need embedding_type param. ")
            return None

        embedding_type = kwargs.pop("embedding_type")

        if not isinstance(embedding_type, str):
            logger.error("embedding_type should be str type. ")
            return None

        if embedding_type not in cls.NPU_SUPPORT_EMB:
            logger.error(f"embedding_type is not support. {embedding_type}")
            return None

        creator = cls.NPU_SUPPORT_EMB.get(embedding_type)

        try:
            embedding = creator(**kwargs)
        except KeyError:
            logger.error(f"create embedding key error")
            return None
        except Exception:
            logger.error(f"exception occurred while constructing embedding")
            return None

        return embedding
