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
MXRAGCache 的similarity 适配器类
"""
from typing import Dict, Tuple, Any

from gptcache.similarity_evaluation import SimilarityEvaluation
from loguru import logger

from mx_rag.reranker.reranker import Reranker
from mx_rag.reranker.reranker_factory import RerankerFactory
from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP


class CacheSimilarity(SimilarityEvaluation):
    """
    功能描述:
        CacheSimilarity 为MXRAG适配gptcache similarity功能的适配器

    Attributes:
        _similarity_impl: (Reranker) 来自MXRAG的reranker实例
        _score_min: (float) 相似度最小值 默认值0
        _score_max: (float) 相似度最大值 默认值1
        _reverse: (bool) 相似度是否取反
    """

    @validate_params(
        similarity=dict(validator=lambda x: isinstance(x, Reranker), message="param must be instance of Reranker"),
        score_min=dict(validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 100.0,
                       message="param must be float or int and value range [0.0, 100.0]"),
        score_max=dict(validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 100.0,
                       message="param must be float or int and value range [0.0, 100.0]"),
        reverse=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, similarity: Reranker, score_min: float = 0.0, score_max: float = 1.0,
                 reverse: bool = False):
        if score_max < score_min:
            raise ValueError("score max must greater than score min")

        self._similarity_impl = similarity
        self._score_min = score_min
        self._score_max = score_max
        self._reverse = reverse

    @staticmethod
    def create(**kwargs):
        """
        构造CacheSimilarity的静态方法

        Args:
            kwargs:(Dict[str, Any]) similarity配置参数
        Return:
            similarity 适配器实例
        """
        score_min = kwargs.pop("score_min", 0.0)
        score_max = kwargs.pop("score_max", 1.0)
        reverse = kwargs.pop("reverse", False)

        similarity = RerankerFactory.create_reranker(**kwargs)
        similarity = CacheSimilarity(similarity, score_min, score_max, reverse)
        return similarity

    def evaluation(
            self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs
    ) -> float:
        """
        进行相似度匹配

        Args:
            src_dict:(Dict[str, Any]) 被比较的数据
            cache_dict:(Dict[str, Any]) 比较的数据
        Return:
            score 比较分数
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]

            if src_question.lower() == cache_question.lower():
                return self._final_result(self._score_max)

            scores = self._similarity_impl.rerank(src_question, [cache_question], batch_size=1)
            return self._final_result(scores[0])
        except KeyError as e:
            logger.error(f"Key error: {e}")
            return self._final_result(self._score_min)
        except Exception as e:
            logger.error(f"CacheSimilarity evaluation fatal error. {e}")
            return self._final_result(self._score_min)

    def range(self) -> Tuple[float, float]:
        return self._score_min, self._score_max

    def _final_result(self, score: float):
        if score > self._score_max:
            score = self._score_max

        if score < self._score_min:
            score = self._score_min

        score = score - self._score_min

        if self._reverse:
            score = (self._score_max - self._score_min - score)
        return score
