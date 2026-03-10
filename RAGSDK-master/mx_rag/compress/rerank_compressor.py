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


from langchain_text_splitters.base import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from mx_rag.compress import PromptCompressor
from mx_rag.reranker import Reranker
from mx_rag.utils.common import validate_params, \
    BOOL_TYPE_CHECK_TIP, MAX_PAGE_CONTENT, STR_TYPE_CHECK_TIP, TEXT_MAX_LEN


class RerankCompressor(PromptCompressor):
    @validate_params(
        reranker=dict(validator=lambda x: isinstance(x, Reranker),
                      message="param must be instance of Reranker"),
        splitter=dict(validator=lambda x: isinstance(x, TextSplitter) or x is None,
                      message="param must be instance of LangChain's TextSplitter or None"),
    )
    def __init__(self,
                 reranker: Reranker,
                 splitter: TextSplitter = None
                 ):
        self.reranker = reranker
        self.splitter = splitter

    @staticmethod
    def _ranked_texts(sentences_list, sorted_idx, compress_rate, context_reorder):
        # 压缩策略：按照排序，优先保留相似性高的句子，直到达到目标
        reserved_ctx_ids = []
        context_sentences_lens = [len(t) for t in sentences_list]
        context_sentences_len_sum = sum(context_sentences_lens)
        target_sentences_len = compress_rate * context_sentences_len_sum

        r_set = set()
        for idx, _ in sorted_idx:
            if idx not in r_set:
                reserved_ctx_ids.append(idx)
                r_set.add(idx)

            target_sentences_len -= context_sentences_lens[idx]
            if target_sentences_len < 0:
                break
        if not context_reorder:
            reserved_ctx_ids = sorted(reserved_ctx_ids)
        compressed_text = ''.join([sentences_list[i] for i in reserved_ctx_ids])
        return compressed_text

    @validate_params(
        context=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PAGE_CONTENT,
                     message=f"param must be str, and length range [1, {MAX_PAGE_CONTENT}]"),
        question=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= TEXT_MAX_LEN,
                      message=STR_TYPE_CHECK_TIP + f", and length range [1, {TEXT_MAX_LEN}]"),
        compress_rate=dict(validator=lambda x: isinstance(x, (float, int)) and 0 < x < 1,
                         message="param must be float or int, and value range (0, 1)"),
        context_reorder=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def compress_texts(self,
                       context: str,
                       question: str,
                       compress_rate: float = 0.6,
                       context_reorder: bool = False):
        if self.splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=0,
                separators=["\n", ""],
                keep_separator=True
            )
            self.splitter = text_splitter
        # 文本切分
        logger.info("Starting text embedding ")
        sentences_list = self.splitter.split_text(text=context)
        # 句子排序
        logger.info("Starting sentence ranking ")
        ranker_result = self.reranker.rerank(query=question, texts=sentences_list)
        sorted_idx = sorted(enumerate(ranker_result), key=lambda x: x[1], reverse=True)
        logger.info("Starting sentence compression ")
        return self._ranked_texts(sentences_list, sorted_idx, compress_rate, context_reorder)
