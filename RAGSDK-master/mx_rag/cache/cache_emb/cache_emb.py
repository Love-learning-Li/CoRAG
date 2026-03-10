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
MXRAGCache 的embedding 适配器类
"""
from typing import List, Union

from langchain_core.embeddings import Embeddings
from gptcache.embedding.base import BaseEmbedding

from mx_rag.embedding import EmbeddingFactory
from mx_rag.utils.common import validate_params, MAX_VEC_DIM, BOOL_TYPE_CHECK_TIP


class CacheEmb(BaseEmbedding):
    """
    功能描述:
        CacheEmb 为MXRAG适配gptcache embedding功能的适配器

    Attributes:
        emb_obj:(Embedding) MXRAG的embedding 实例
        x_dim: (int) embedding的维度
        skip_emb: (bool) 是否需要跳过embedding 对于memory_cache 不需要做embedding
    """

    @validate_params(
        emb_obj=dict(validator=lambda x: isinstance(x, Embeddings) or x is None,
                     message="param must be instance of Embeddings or None"),
        x_dim=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_VEC_DIM,
                   message="param must be int and value range [0, 1024 * 1024]"),
        skip_emb=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, emb_obj: Embeddings = None, x_dim: int = 0, skip_emb: bool = False):
        self.emb_obj = emb_obj
        self.x_dim = x_dim
        self.skip_emb = skip_emb

    @staticmethod
    def create(**kwargs):
        """
        构造CacheEmb的静态方法

        Args:
            kwargs:(Dict[str, Any]) embedding配置参数
        Return:
            embedding 适配器实例
        """
        x_dim = kwargs.pop("x_dim", 0)
        skip_emb = kwargs.pop("skip_emb", False)

        embedding = EmbeddingFactory.create_embedding(**kwargs)
        embedding = CacheEmb(embedding, x_dim, skip_emb)
        return embedding

    def to_embeddings(self, data: Union[List[str], str], **kwargs):
        """
        调用MXRAG的embedding方法去embedding 来自gptcache的数据data

        Args:
            data:(List[str]) 需要被embedding的数据
        Return:
            embedding之后的数据, 如果是跳过embedding 则返回原始数据
        """
        if self.skip_emb:
            return data

        if isinstance(data, str):
            data = [data]
        return self._embedding_text(data)

    def dimension(self) -> int:
        """
        由GPTCache 调用，返回embedding模块的维度信息

        Return:
            返回embedding的维度，由gptcache框架调用
        """
        return self.x_dim if not self.skip_emb else 0

    def _embedding_text(self, data: List[str]) -> List[List[float]]:
        """
        调用MXRAG的embedding方法去embedding 来自gptcache的数据data

        Return:
            返回embedding 之后的数据
        """
        if not isinstance(self.emb_obj, Embeddings):
            raise TypeError("emb_obj is not instance of Embeddings")

        return self.emb_obj.embed_documents(data)
