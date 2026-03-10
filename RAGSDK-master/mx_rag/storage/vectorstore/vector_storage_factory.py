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
向量数据库的工厂类，用于生产mxrag的向量数据库
"""
from abc import ABC
from typing import Optional
from loguru import logger

from mx_rag.storage.vectorstore import VectorStore, OpenGaussDB
from mx_rag.storage.vectorstore import MilvusDB


class VectorStorageError(Exception):
    """
    向量数据库错误
    """
    pass


try:
    import ascendfaiss
    from mx_rag.storage.vectorstore import MindFAISS
    MIND_FAISS_AVAILABLE = True
except ImportError as e:
    MIND_FAISS_AVAILABLE = False


class VectorStorageFactory(ABC):
    """
    功能描述:
        向量数据库的工厂方法类，用于生产mxrag的向量数据库

    Attributes:
        _NPU_SUPPORT_VEC_TYPE 字典，用于映射向量数据库和对应的构造函数
    """
    _NPU_SUPPORT_VEC_TYPE = {
        "opengauss_db": OpenGaussDB.create,
        "milvus_db": MilvusDB.create
    }
    if MIND_FAISS_AVAILABLE:
        _NPU_SUPPORT_VEC_TYPE["npu_faiss_db"] = MindFAISS.create


    @classmethod
    def create_storage(cls, **kwargs) -> Optional[VectorStore]:
        """
        功能描述:
            构造vector storage

        Args:
            kwargs: Dict[str, Any] 构造向量数据库的参数
        Return:
            vector_store: VectorStore 返回的构造向量数据库实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "vector_type" not in kwargs:
            raise VectorStorageError("The 'vector_type' parameter is required.")

        vector_type = kwargs.pop("vector_type")

        if not isinstance(vector_type, str):
            raise VectorStorageError("The 'vector_type' parameter must be of type str.")

        if vector_type not in cls._NPU_SUPPORT_VEC_TYPE:
            raise VectorStorageError(f"The specified 'vector_type' '{vector_type}' is not supported.")

        creator = cls._NPU_SUPPORT_VEC_TYPE.get(vector_type)
        try:
            vector_store = creator(**kwargs)
        except KeyError as e:
            raise VectorStorageError("A KeyError occurred while creating the vector store.") from e
        except Exception as e:
            raise VectorStorageError("An unexpected error occurred while constructing the vector store.") from e

        return vector_store
