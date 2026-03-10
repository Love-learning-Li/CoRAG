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

from typing import List, Callable, Union

from langchain_text_splitters.base import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import torch
import numpy as np
from loguru import logger

from mx_rag.compress import PromptCompressor
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP, MAX_PAGE_CONTENT, MAX_DEVICE_ID, TEXT_MAX_LEN, \
    MAX_CHUNKS_NUM


class ClusterCompressor(PromptCompressor):
    @validate_params(
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        embed=dict(validator=lambda x: isinstance(x, Embeddings),
                   message="param must be instance of LangChain's Embeddings"),
        cluster_func=dict(validator=lambda x: isinstance(x, Callable),
                          message="param must be Callable[[List[List[float]]], Union[List[int], np.ndarray]] function"),
        splitter=dict(validator=lambda x: isinstance(x, TextSplitter) or x is None,
                      message="param must be instance of LangChain's TextSplitter or None"),
    )
    def __init__(self,
                 cluster_func: Callable[[List[List[float]]], Union[List[int], np.ndarray]],
                 embed: Embeddings,
                 splitter: TextSplitter = None,
                 dev_id: int = 0,
                 ):
        self.embed = embed
        self.cluster_func = cluster_func
        self.splitter = splitter
        self.dev_id = dev_id

    @staticmethod
    def _assemble_result(sentences, labels, similarity, compress_rate):
        # 根据压缩率，每个社区删除对应的比例，相似性差的先删
        reserved_sentences = []
        community = {}
        for index, label in enumerate(labels):
            if label not in community:
                community[label] = [index]
            else:
                community[label].append(index)
        for _, value in community.items():
            similarity_temp = [similarity[i] for i in value]
            sorted_sentences = np.argsort(similarity_temp)
            reserved_index = sorted_sentences[int(len(value) * compress_rate):]
            for left_index in reserved_index:
                reserved_sentences.append(value[left_index])

        reserved_sentences = sorted(reserved_sentences)
        compress_context = ''.join([sentences[i] for i in reserved_sentences])
        return compress_context

    @validate_params(
        context=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PAGE_CONTENT,
                     message=f"param must be str, and length range [1, {MAX_PAGE_CONTENT}]"),
        question=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= TEXT_MAX_LEN,
                      message=STR_TYPE_CHECK_TIP + f", and length range [1, {TEXT_MAX_LEN}]"),
        compress_rate=dict(validator=lambda x: isinstance(x, (float, int)) and 0 < x < 1,
                           message=f"param must be float or int and value range (0, 1)"),
    )
    def compress_texts(self,
                       context: str,
                       question: str,
                       compress_rate: float = 0.6,
                       ):
        if self.splitter is None:
            sentence_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=0,
                separators=["。", "！", "？", "\n", "，", "；", " ", ""],  # 中文分隔符列表
            )
            self.splitter = sentence_splitter
        # 文本切分
        logger.info("Starting text splitting ")
        sentences = self.splitter.split_text(text=context)

        if len(sentences) < 2:
            return context
        # 文本embedding
        logger.info("Starting text embedding ")
        sentences_with_question = sentences + [question]
        sentences_embedding_with_question = self.embed.embed_documents(sentences_with_question)
        sentences_embedding = sentences_embedding_with_question[:-1]
        question_embedding = sentences_embedding_with_question[-1]
        # 计算余弦相似度
        logger.info("Starting cosine calculate similarity ")
        similarity = np.array(self._get_similarity(sentences_embedding, question_embedding).to('cpu'))
        # 社区划分
        logger.info("Starting community division ")
        label = self.cluster_func(sentences_embedding)
        if not isinstance(label, (np.ndarray, list)):
            raise TypeError(f"callback function {self.cluster_func.__name__} "
                            f"returned invalid result, must be List[int] or np.ndarray")
        if len(label) > MAX_CHUNKS_NUM:
            raise ValueError(f"callback function {self.cluster_func.__name__} "
                             f"returned invalid result, length exceeds {MAX_CHUNKS_NUM}.")
        if not len(label) == len(sentences):
            raise ValueError(f"callback function {self.cluster_func.__name__} returned invalid result."
                             f" length must match sentences length {len(sentences)}.")
        # 压缩文本
        logger.info("Starting text compression ")
        compress_context = self._assemble_result(sentences, label, similarity, compress_rate)

        return compress_context

    def _get_similarity(self, sentences_embedding, question_embedding):
        # 计算句子和问题的相似度
        sentences_embedding = np.array(sentences_embedding, dtype=np.float32)
        question_embedding = np.array(question_embedding, dtype=np.float32)

        c1 = torch.from_numpy(sentences_embedding).to(f'npu:{self.dev_id}')
        c2 = torch.nn.functional.normalize(c1, p=2, dim=-1)

        q1 = torch.from_numpy(question_embedding).to(f'npu:{self.dev_id}')
        q2 = torch.nn.functional.normalize(q1, p=2, dim=-1)

        sims_with_query = q2.squeeze() @ c2.T  # 余弦相似度
        return sims_with_query
