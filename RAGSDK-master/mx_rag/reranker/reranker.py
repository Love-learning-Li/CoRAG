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


from abc import ABC, abstractmethod
from typing import List

import numpy as np
from langchain_core.documents import Document

from mx_rag.utils.common import (validate_params, validate_list_document, TEXT_MAX_LEN, STR_MAX_LEN, validate_list_str)


class Reranker(ABC):
    def __init__(self, k: int):
        self._k = k

    @abstractmethod
    def rerank(self,
               query: str,
               texts: List[str],
               batch_size: int = 32):
        """ rank the texts and query"""

    @validate_params(
        scores=dict(validator=lambda x: isinstance(x, np.ndarray) and x.ndim == 1 and 1 <= len(x) <= TEXT_MAX_LEN,
                    message="np.array length range [1, 1000 * 1000]")
    )
    def rerank_top_k(self,
                     objs: List,
                     scores: np.ndarray) -> List:
        check_objs_flag = False
        if isinstance(objs, list) and len(objs) > 0:
            if isinstance(objs[0], str) and validate_list_str(objs, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                check_objs_flag = True
            if isinstance(objs[0], Document) and validate_list_document(objs, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                check_objs_flag = True
        if not check_objs_flag:
            raise ValueError(f"param objs of function rerank_top_k must meets: Type is List[Document] or List[str], "
                             f"list length range [1, {TEXT_MAX_LEN}], "
                             f"str length range [1, {STR_MAX_LEN}]")

        if not len(objs) == len(scores):
            raise ValueError(f"objs and scores expected to be equal length")

        obj_scores = list(zip(objs, scores))
        obj_scores.sort(reverse=True, key=lambda ele: ele[1])

        res = [obj_score[0] for obj_score in obj_scores]
        return res[:self._k]
