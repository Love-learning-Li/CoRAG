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


from loguru import logger
from tqdm import tqdm

from mx_rag.reranker import Reranker
from mx_rag.utils.common import validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN


@validate_params(
    reranker=dict(validator=lambda x: isinstance(x, Reranker),
                  message="param must be instance of Reranker"),
    query_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                    message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                            f"str length range [1, {STR_MAX_LEN}]"),
    doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                          f"str length range [1, {STR_MAX_LEN}]")
)
def reranker_featured(reranker: Reranker, query_list: list[str], doc_list: list[str]):
    """重排模型打分"""
    if len(query_list) != len(doc_list):
        logger.error(f"reranker_featured query_list and doc_list has different len")
        return []

    score_list = []

    for query, doc in tqdm(zip(query_list, doc_list), total=len(query_list)):
        scores = reranker.rerank(query, [doc])
        score_list.append(scores[0] if scores.size > 0 else 0)

    return score_list
