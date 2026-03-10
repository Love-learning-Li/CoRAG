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

from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

import numpy as np
from loguru import logger

from mx_rag.utils.common import MAX_FILTER_SEARCH_ITEM, MAX_STDOUT_STR_LEN, validate_params


class SearchMode(Enum):
    DENSE = 0  # dense search
    SPARSE = 1  # sparse search
    HYBRID = 2  # hybrid search


class VectorStore(ABC):
    MAX_VEC_NUM = 100 * 1000 * 1000 * 1000
    MAX_SEARCH_BATCH = 1024 * 1024

    def __init__(self):
        self.score_scale = None

    @abstractmethod
    def delete(self, ids):
        pass

    @abstractmethod
    def search(self, embeddings, k, filter_dict=None):
        pass

    @abstractmethod
    def add(self, ids, embeddings, document_id):
        pass

    @abstractmethod
    def add_sparse(self, ids, sparse_embeddings):
        pass

    @abstractmethod
    def add_dense_and_sparse(self, ids, dense_embeddings, sparse_embeddings):
        pass

    @abstractmethod
    def get_all_ids(self):
        pass

    @abstractmethod
    def update(self, ids: List[int], dense: Optional[np.ndarray] = None,
               sparse: Optional[List[Dict[int, float]]] = None):
        pass

    @validate_params(
        threshold=dict(
            validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 1.0,
            message="param must be float or int and value range [0.0, 1.0]",
        )
    )
    def search_with_threshold(self, embeddings: Union[List[List[float]], List[Dict[int, float]]],
                              k: int = 3, threshold: float = 0.1, filter_dict=None):
        """
        根据阈值进行查找，过滤掉不满足的分数
        Args:
            filter_dict: 检索的过滤条件
            embeddings: 词嵌入之后的查询
            k: top_k个结果
            threshold: 阈值

        Returns: 通过search过滤之后的分数

        """
        scores, indices = self.search(embeddings, k, filter_dict=filter_dict)[:2]

        logger.info(f"threshold is [>={threshold}]")

        filter_score = []
        filter_indices = []
        for i, score in enumerate(scores[0]):
            if score >= threshold:
                filter_score.append(scores[0][i])
                filter_indices.append(indices[0][i])

        return [filter_score], [filter_indices]

    def as_retriever(self, **kwargs):
        """
        向量数据库转换为向量检索器
        Args:
            **kwargs:

        Returns: Retriever

        """
        from mx_rag.retrievers.retriever import Retriever

        return Retriever(vector_store=self, **kwargs)

    def save_local(self):
        pass

    def get_save_file(self):
        return ""

    def get_ntotal(self) -> int:
        return 0

    def _score_scale(self, scores: List[List[float]]) -> List[List[float]]:
        """
        分数量化
        Args:
            scores: 词嵌入的得分

        Returns: 量化之后的分数

        """
        if self.score_scale is not None:
            scores = [[self.score_scale(x) for x in row] for row in scores]
        return scores

    def _validate_filter_dict(self, filter_dict):
        if not filter_dict:
            return
        if len(filter_dict) > MAX_FILTER_SEARCH_ITEM:
            raise ValueError(
                f"filter_dict invalid length({len(filter_dict)}) is greater than {MAX_FILTER_SEARCH_ITEM}")
        invalid_keys = str(filter_dict.keys() - {"document_id"})
        if invalid_keys:
            logger.warning(f"{invalid_keys[:MAX_STDOUT_STR_LEN]} ... is no support")
        doc_filter = filter_dict.get("document_id", [])
        if not isinstance(doc_filter, list) or not all(isinstance(item, int) for item in doc_filter):
            raise ValueError("value of 'document_id' in filter_dict must be List[int]")
        doc_filter = list(set(doc_filter))  # 去重
        max_ids_len = len(self.get_all_ids())
        if len(doc_filter) > max_ids_len:
            raise ValueError(f"length of 'document_id' in filter_dict over than length of ids({max_ids_len})")
