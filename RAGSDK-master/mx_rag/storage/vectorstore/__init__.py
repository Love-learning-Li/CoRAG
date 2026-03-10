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


__all__ = [
    "VectorStore",
    "MilvusDB",
    "OpenGaussDB",
    "VectorStorageFactory",
    "SearchMode"
]

try:
    from .faiss_npu import MindFAISS
    __all__.append("MindFAISS")
except ImportError as e:
    MindFAISS = None

from .vectorstore import VectorStore, SearchMode
from .milvus import MilvusDB
from .opengauss import OpenGaussDB
from .vector_storage_factory import VectorStorageFactory
