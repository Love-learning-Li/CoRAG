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

import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from loguru import logger

from mx_rag.utils.common import validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN, BOOL_TYPE_CHECK_TIP


@validate_params(
    query_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                    message="param must meets: Type is List[str], "
                            f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]"),
    doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message="param must meets: Type is List[str], "
                          f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]"),
    use_quick=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
)
def bm25_featured(query_list: list[str], doc_list: list[str], use_quick: bool = True):
    """bm25对文档对打分"""

    if len(query_list) != len(doc_list):
        logger.error(f"bm25_featured query_list and doc_list has different len")
        return []

    def chinese_tokenizer(text):
        return list(jieba.cut(text))

    score_list = []
    if use_quick:
        # 记录每个查询词对应的原始文档索引,保证query不重复
        query_to_doc_index = {query: i for i, query in enumerate(query_list)}

        # 去重文档
        seen = set()
        unique_docs = [x for x in doc_list if x not in seen and not seen.add(x)]

        # 构建 BM25 模型
        tokenized_docs = [chinese_tokenizer(doc) for doc in unique_docs]
        bm25 = BM25Okapi(tokenized_docs)

        # 计算 BM25 得分，并记录每个查询词对应的原始文档索引
        for query, doc_index in tqdm(query_to_doc_index.items(), desc="bm25 sort quick",
                                     disable=len(query_to_doc_index) < 128):
            tokenized_query = chinese_tokenizer(query)
            scores_for_query = bm25.get_scores(tokenized_query)

            doc = doc_list[doc_index]
            unique_doc_index = unique_docs.index(doc)
            score_list.append(scores_for_query[unique_doc_index])
    else:
        # 对每个文档进行分词
        tokenized_doc = [chinese_tokenizer(doc) for doc in doc_list]
        bm25 = BM25Okapi(tokenized_doc)
        for index, query in enumerate(tqdm(query_list, desc="bm25 sort", disable=len(query_list) < 128)):
            tokenized_query = chinese_tokenizer(query)
            doc_scores = bm25.get_scores(tokenized_query)
            score_list.append(doc_scores[index])

    return score_list
