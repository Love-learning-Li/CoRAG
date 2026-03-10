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

import os
from typing import Callable

from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.reranker import Reranker
from mx_rag.tools.finetune.dataprocess.generate_qd import GENERATE_QD_PROMPT
from mx_rag.tools.finetune.dataprocess.improve_query import improve_query
from mx_rag.tools.finetune.dataprocess.llm_preferred import SCORING_QD_PROMPT
from mx_rag.tools.finetune.generator.common import BaseGenerator
from mx_rag.utils.common import (validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN,
                                 MAX_PATH_LENGTH, MAX_PROMPT_LENGTH, BOOL_TYPE_CHECK_TIP)
from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.file_operate import write_jsonl_to_file, read_jsonl_from_file

MAX_DATASET_LEN = 10000


class DataProcessConfig:
    """
    功能描述:
        DataProcessConfig 微调合成数据生成配置

    Attributes:
        generate_qd_prompt: (str) 自动生成QD对的LLM所用Prompt
        llm_preferred_prompt: (str) 通过LLM优选QD对的Prompt
        question_number: (int) 针对每个原始doc生成的query问题数
        featured: (bool) QD对精选开关, 借助BM25和reranker融合排序筛选
        featured_percentage: (float) 精选筛选比例
        preferred: (bool) QD对优选开关, 借助LLM进行评分筛选
        llm_threshold_score: (float) 优选筛选比例
        rewrite: (bool) 问题重写开关
        query_rewrite_number: (int) 问题重写的数量
    """

    @validate_params(
        generate_qd_prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                                message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
        llm_preferred_prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                                  message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
        question_number=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 20,
                             message="param must meets: Type is int, length range (0, 20]"),
        featured=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        featured_percentage=dict(validator=lambda x: isinstance(x, (float, int)) and 0.0 < x < 1.0,
                                 message="param must be float or int and value range (0.0, 1.0)"),
        preferred=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        llm_threshold_score=dict(validator=lambda x: isinstance(x, (float, int)) and 0.0 < x < 1.0,
                                 message="param must be float or int and value range (0.0, 1.0)"),
        rewrite=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        query_rewrite_number=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 20,
                                  message="param must meets: Type is int, length range (0, 20]"),
    )
    def __init__(self,
                 generate_qd_prompt: str = GENERATE_QD_PROMPT,
                 llm_preferred_prompt: str = SCORING_QD_PROMPT,
                 question_number: int = 3,
                 featured: bool = True,
                 featured_percentage: float = 0.8,
                 preferred: bool = True,
                 llm_threshold_score: float = 0.8,
                 rewrite: bool = True,
                 query_rewrite_number: int = 2):
        self.generate_qd_prompt = generate_qd_prompt
        self.llm_preferred_prompt = llm_preferred_prompt
        self.question_number = question_number
        self.featured = featured
        self.featured_percentage = featured_percentage
        self.preferred = preferred
        self.llm_threshold_score = llm_threshold_score
        self.rewrite = rewrite
        self.query_rewrite_number = query_rewrite_number


class TrainDataGenerator(BaseGenerator):
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        dataset_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PATH_LENGTH,
                          message=f"param must be str and str length range (0, {MAX_PATH_LENGTH}]"),
        reranker=dict(validator=lambda x: isinstance(x, Reranker),
                      message="param must be instance of Reranker"),
        encrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="encrypt_fun must be None or callable function"),
        decrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="decrypt_fun must be None or callable function")
    )
    def __init__(self, llm: Text2TextLLM, dataset_path: str, reranker: Reranker,
                 encrypt_fn: Callable[[str], str] = None, decrypt_fn: Callable[[str], str] = None):
        super().__init__(llm, dataset_path, encrypt_fn, decrypt_fn)
        self.reranker = reranker

    @validate_params(
        split_doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                            message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                                    f"str length range [1, {STR_MAX_LEN}]"),
        data_process_config=dict(validator=lambda x: isinstance(x, DataProcessConfig),
                                 message="param must be instance of DataProcessConfig"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1024,
                        message="param must meets: Type is int, length range (0, 1024]"),
    )
    def generate_train_data(self,
                            split_doc_list: list[str],
                            data_process_config: DataProcessConfig,
                            batch_size: int = 8
                            ):
        FileCheck.dir_check(self.dataset_path)
        corpus_data_path = os.path.join(self.dataset_path, "train_data.jsonl")
        if os.path.exists(corpus_data_path):
            logger.info("embedding train data has been created.")
            return

        # 流程开始
        logger.info("step Generating rough problem documentation pairs")
        query_list, doc_list = self._generate_coarsest_qd_pairs(split_doc_list, data_process_config.question_number,
                                                                data_process_config.generate_qd_prompt,
                                                                batch_size)
        logger.info("step Generated rough problem documentation pairs finished")

        if data_process_config.featured:
            logger.info("step bm25+reranker query document pair selection")
            query_list, doc_list = self._feature_qd_pair(query_list,
                                                         doc_list,
                                                         self.reranker,
                                                         data_process_config.featured_percentage)
            logger.info("step bm25+reranker selection finished")

        if data_process_config.preferred:
            logger.info("step LLM optimizing query document pair")
            query_list, doc_list = self._prefer_qd_pair(query_list,
                                                        doc_list,
                                                        data_process_config.llm_threshold_score,
                                                        data_process_config.llm_preferred_prompt,
                                                        batch_size)
            logger.info("step LLM optimizing query document pair finished")

        if data_process_config.rewrite:
            logger.info("step Enhancing query diversity and preserving training data")
            query_list, doc_list = self._rewrite_query(query_list,
                                                       doc_list,
                                                       data_process_config.query_rewrite_number,
                                                       batch_size)
            logger.info("step Enhancing query diversity and preserving training data finished")

        train_data = []
        for query, doc in zip(query_list, doc_list):
            train_data.append({"query": self._encrypt(query), "corpus": self._encrypt(doc)})

        train_data_path = os.path.join(self.dataset_path, "train_data.jsonl")
        write_jsonl_to_file(train_data, train_data_path)

        return

    def _rewrite_query(self,
                       preferred_query_list: list[str],
                       preferred_doc_list: list[str],
                       query_rewrite_number: int,
                       batch_size: int):
        if len(preferred_query_list) > MAX_DATASET_LEN or len(preferred_doc_list) > MAX_DATASET_LEN:
            logger.error(f"inputs len should not bigger than {MAX_DATASET_LEN}")
            return [], []

        if len(preferred_query_list) != len(preferred_doc_list):
            logger.error(f"preferred_query_list and preferred_doc_list has different len")
            return [], []

        FileCheck.dir_check(self.dataset_path)

        query_list = []
        doc_list = []
        query_list.extend(preferred_query_list)
        doc_list.extend(preferred_doc_list)

        rewrite_data_path = os.path.join(self.dataset_path, "rewrite_data.jsonl")
        if os.path.exists(rewrite_data_path):
            rewrite_data_list = read_jsonl_from_file(rewrite_data_path)
            for rewrite_data in rewrite_data_list:
                query_list.append(self._decrypt(rewrite_data["query"]))
                doc_list.append(self._decrypt(rewrite_data["corpus"]))
            if len(query_list) == len(preferred_query_list) * query_rewrite_number:
                logger.info("rewrite query finished, skip rewrite query process")
                return query_list, doc_list
            else:
                # 根据重写的长度计算还需要重写的数据量
                if len(preferred_query_list) == 0:
                    per_rewrite_number = 0
                    remain_number = query_rewrite_number - 1
                else:
                    rewritten_number = int(len(rewrite_data_list) / len(preferred_query_list))
                    per_rewrite_number = len(rewrite_data_list) % len(preferred_query_list)
                    remain_number = query_rewrite_number - rewritten_number - 1
                if remain_number >= 0:
                    logger.info("rewrite query not finished, continue to rewrite query process")
                    remain_queries = preferred_query_list[per_rewrite_number:] + preferred_query_list * remain_number
                    remain_docs = preferred_doc_list[per_rewrite_number:] + preferred_doc_list * remain_number
                    new_query_list = self._rewrite(remain_queries, remain_docs, rewrite_data_path, batch_size)
                    query_list.extend(new_query_list)
                    doc_list.extend(remain_docs)
                else:
                    logger.info('calculated based on parameters, do not need to rewrite the query')
        else:
            for i in range(query_rewrite_number):
                logger.info(f"The {i + 1}st times rewrite the query")
                new_query_list = self._rewrite(preferred_query_list, preferred_doc_list, rewrite_data_path, batch_size)
                query_list.extend(new_query_list)
                doc_list.extend(preferred_doc_list)

        return query_list, doc_list

    def _rewrite(self, preferred_query_list, preferred_doc_list, rewrite_data_path, batch_size):
        logger.info(f"rewrite query count: {len(preferred_query_list)}")

        query_list = []
        count = 0
        for i in range(0, len(preferred_query_list), batch_size):
            chunk_query_list = preferred_query_list[i:i + batch_size]
            chunk_doc_list = preferred_doc_list[i:i + batch_size]
            new_query_list = improve_query(self.llm, chunk_query_list)
            temp_qd_pairs = []
            for query, doc in zip(new_query_list, chunk_doc_list):
                if query != "":
                    query_list.append(query)
                    temp_qd_pairs.append({"query": self._encrypt(query), "corpus": self._encrypt(doc)})
            write_jsonl_to_file(temp_qd_pairs, rewrite_data_path, 'a')
            logger.info(f"The {count + 1} st time rewrite query success by chunk {batch_size}")
            count += 1

        return query_list
