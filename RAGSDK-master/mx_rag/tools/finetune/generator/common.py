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
from pathlib import Path
from typing import Callable, List

from loguru import logger

from mx_rag.document import LoaderMng
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker import Reranker
from mx_rag.tools.finetune.dataprocess.bm25_featured import bm25_featured
from mx_rag.tools.finetune.dataprocess.generate_qd import generate_qa_embedding_pairs
from mx_rag.tools.finetune.dataprocess.llm_preferred import llm_preferred
from mx_rag.tools.finetune.dataprocess.reciprocal_rank_fusion import reciprocal_rank_fusion
from mx_rag.tools.finetune.dataprocess.reranker_featured import reranker_featured
from mx_rag.utils.common import validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN, validate_params, MAX_PATH_LENGTH, \
    CALLABLE_TYPE_CHECK_TIP
from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.utils.file_operate import read_jsonl_from_file, write_jsonl_to_file

MAX_DATASET_LEN = 10000
MAX_FILE_SIZE_100M = 100 * 1024 * 1024
MAX_FILE_PROCESS_TIMES = 1000
SAMPLE_RANGE_MIN = 100


class BaseGenerator:

    def __init__(self, llm: Text2TextLLM, dataset_path: str, encrypt_fn: Callable[[str], str] = None,
                 decrypt_fn: Callable[[str], str] = None):
        self.llm = llm
        self.dataset_path = dataset_path
        self.encrypt_fn = encrypt_fn
        self.decrypt_fn = decrypt_fn

    @validate_params(
        document_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PATH_LENGTH,
                           message=f"param must be str and str length range (0, {MAX_PATH_LENGTH}]"),
        loader_mng=dict(validator=lambda x: isinstance(x, LoaderMng), message="param must be instance of LoaderMng"),
        filter_func=dict(validator=lambda x: x is None or isinstance(x, Callable), message=CALLABLE_TYPE_CHECK_TIP),
    )
    def generate_origin_document(self, document_path: str, loader_mng: LoaderMng,
                                 filter_func: Callable[[List[str]], List[str]] = None):
        logger.info("Original document splitting")
        FileCheck.dir_check(document_path)

        def doc_load(file_path: str):
            SecFileCheck(file_path, MAX_FILE_SIZE_100M).check()

            doc_type = os.path.splitext(file_path)[-1]

            loader_info = loader_mng.get_loader(doc_type)
            loader = loader_info.loader_class(file_path=file_path, **loader_info.loader_params)

            docs = []
            for doc in loader.load():
                splitter_info = loader_mng.get_splitter(doc_type)
                splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
                docs.extend(splitter.split_documents([doc]))

            return docs

        # 对数据进行清洗
        def execute_callback(split_texts: List[str]):
            if isinstance(filter_func, Callable):
                filter_texts = filter_func(split_texts)
                if not validate_list_str(filter_texts, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                    logger.error(f"The return value of the callback method is not List[str], use raw doc slice")
                    return split_texts
                else:
                    return filter_texts
            return split_texts

        doc_cnt = 0
        split_doc_list = []
        doc_set = set()

        for file_type in loader_mng.loaders:
            logger.info(f"load {file_type} file")
            for doc_file in Path(document_path).glob(f"*{file_type}"):
                if doc_cnt > MAX_FILE_PROCESS_TIMES:
                    logger.warning(f"unable to process files over {MAX_FILE_PROCESS_TIMES} times")
                    break
                if not doc_file.is_file():
                    continue

                for doc in doc_load(doc_file.as_posix()):
                    texts = execute_callback([doc.page_content])
                    # 去重
                    unique_docs = [x for x in texts if x not in doc_set and not doc_set.add(x)]
                    split_doc_list.extend(unique_docs)
                doc_cnt = doc_cnt + 1
        if doc_cnt == 0:
            logger.warning("no valid file found, please check your file type")

        return split_doc_list

    def _generate_coarsest_qd_pairs(self,
                                    split_doc_list: list[str],
                                    question_number: int,
                                    prompt: str,
                                    batch_size):
        logger.info("query document pair generation")

        query_list = []
        doc_list = []
        origin_train_data_path = os.path.join(self.dataset_path, "origin_train_data.jsonl")
        if not os.path.exists(origin_train_data_path):
            query_list, doc_list = self._generate_qd_pairs(
                split_doc_list, question_number, origin_train_data_path, prompt, batch_size
            )
        else:
            logger.info("The qd file is existed, check whether the next generation is required.")
            qd_pairs = read_jsonl_from_file(origin_train_data_path)

            for qd in qd_pairs:
                query_list.append(self._decrypt(qd["query"]))
                doc_list.append(self._decrypt(qd["corpus"]))
            interrupted = doc_list[-1]
            interrupted_index = split_doc_list.index(interrupted)
            if interrupted_index == len(split_doc_list) - 1:
                logger.info("qd pairs generate finished, skip the generation process")
            else:
                logger.info("qd pairs generate not finished, continue to process")
                remain_doc_list = split_doc_list[(interrupted_index + 1):]
                new_query_list, new_doc_list = self._generate_qd_pairs(
                    remain_doc_list, question_number, origin_train_data_path, prompt, batch_size
                )
                query_list.extend(new_query_list)
                doc_list.extend(new_doc_list)

        # 去重
        deduplicate_seen = set()
        deduplicate_queries = []
        deduplicate_docs = []
        for query, doc in zip(query_list, doc_list):
            if query not in deduplicate_seen:
                deduplicate_seen.add(query)
                deduplicate_queries.append(query)
                deduplicate_docs.append(doc)
        logger.info(f'remove duplicate queries len is {len(deduplicate_queries)}')
        return deduplicate_queries, deduplicate_docs

    def _feature_qd_pair(self, query_list: list[str], doc_list: list[str],
                         reranker: Reranker, featured_percentage: float):
        """文档精选，使用bm25和reranker共同打分，按比率保留前面的问答对"""
        if not (1 > featured_percentage > 0):
            raise ValueError("featured_percentage must 0 ~ 1 range")
        logger.info("Selection-bm25 Scoring")
        bm25_scores_path = os.path.join(self.dataset_path, 'bm25_scores.jsonl')
        bm25_scores = []
        if os.path.exists(bm25_scores_path):
            datas = read_jsonl_from_file(bm25_scores_path)
            if len(datas) == len(query_list):
                bm25_scores = [data['score'] for data in datas]
        if len(bm25_scores) == 0:
            bm25_scores = bm25_featured(query_list, doc_list)
            # 保存bm25打分的分数
            datas = [{'query': self._encrypt(query), 'corpus': self._encrypt(doc), 'score': score}
                     for query, doc, score in zip(query_list, doc_list, bm25_scores)]
            write_jsonl_to_file(datas, bm25_scores_path)
        bm25_sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_query_list = [query_list[i] for i in bm25_sorted_indices]

        logger.info("Selection-reranker Scoring")
        reranker_scores_path = os.path.join(self.dataset_path, 'reranker_scores.jsonl')
        reranker_scores = []
        if os.path.exists(reranker_scores_path):
            datas = read_jsonl_from_file(reranker_scores_path)
            if len(datas) == len(query_list):
                reranker_scores = [data['score'] for data in datas]
        if len(reranker_scores) == 0:
            reranker_scores = reranker_featured(reranker, query_list, doc_list)
            # 保存reranker打分的分数
            datas = [{'query': self._encrypt(query), 'corpus': self._encrypt(doc), 'score': score}
                     for query, doc, score in zip(query_list, doc_list, reranker_scores)]
            write_jsonl_to_file(datas, reranker_scores_path)
        reranker_sorted_indices = sorted(range(len(reranker_scores)), key=lambda i: reranker_scores[i], reverse=True)
        reranker_query_list = [query_list[i] for i in reranker_sorted_indices]

        logger.info("RRF algorithm fuses two sorting results")
        fused_query_list = reciprocal_rank_fusion([bm25_query_list, reranker_query_list])

        # 将两个列表打包成元组列表
        zipped_lists = list(zip(query_list, doc_list))

        # 自定义排序函数，根据排序列表的顺序对元组进行排序
        def custom_sort(item):
            return fused_query_list.index(item[0])

        # 根据自定义排序函数对元组列表进行排序
        sorted_zipped_lists = sorted(zipped_lists, key=custom_sort)
        # 将排序后的元组列表解包并拆解成两个可迭代对象
        sorted_query_list, sorted_doc_list = zip(*sorted_zipped_lists)

        logger.info(f"Select the top {featured_percentage * 100}% data as the featured set based "
                    f"on the set parameters")
        featured_query_list = list(sorted_query_list[:round(len(sorted_query_list) * featured_percentage)])
        featured_doc_list = list(sorted_doc_list[:round(len(sorted_doc_list) * featured_percentage)])

        return featured_query_list, featured_doc_list

    def _prefer_qd_pair(self, featured_query_list: list[str], featured_doc_list: list[str],
                        llm_threshold_score: float, prompt: str, batch_size: int = 512):
        """大模型精选"""
        if not (1 > llm_threshold_score > 0):
            raise ValueError("featured_percentage must 0 ~ 1 range")
        logger.info("LLM score and eliminate those whose scores are lower than the preset threshold")
        llm_scores_path = os.path.join(self.dataset_path, 'llm_scores.jsonl')
        llm_scores = []
        if os.path.exists(llm_scores_path):
            scored_data_list = read_jsonl_from_file(llm_scores_path)
            scored_query_list = [self._decrypt(data['query']) for data in scored_data_list]

            interrupted = scored_query_list[-1]
            interrupted_index = featured_query_list.index(interrupted)
            llm_scores = [data['score'] for data in scored_data_list]
            if interrupted_index == len(featured_query_list) - 1:
                logger.info("LLM scoring finished, skip the LLM scoring process")
            else:
                logger.info("LLM scoring not finished, continue to LLM scoring process")
                remain_query_list = featured_query_list[(interrupted_index + 1):]
                remain_doc_list = featured_doc_list[(interrupted_index + 1):]
                scores = self._prefer_scoring(remain_query_list, remain_doc_list, llm_scores_path, prompt, batch_size)
                llm_scores.extend(scores)
        else:
            llm_scores = self._prefer_scoring(
                featured_query_list, featured_doc_list, llm_scores_path, prompt, batch_size
            )
        # 使用列表推导式筛选出所有低于阈值分数的数据，并统计筛选结果的长度
        count_upper_threshold_score = len([x for x in llm_scores if x >= llm_threshold_score])

        llm_sorted_indices = sorted(range(len(llm_scores)), key=lambda i: llm_scores[i], reverse=True)
        llm_query_list = [featured_query_list[i] for i in llm_sorted_indices]
        llm_doc_list = [featured_doc_list[i] for i in llm_sorted_indices]

        preferred_query_list = llm_query_list[:count_upper_threshold_score]
        preferred_doc_list = llm_doc_list[:count_upper_threshold_score]

        return preferred_query_list, preferred_doc_list

    def _prefer_scoring(self, query_list, doc_list, llm_scores_path, prompt, batch_size: int):
        logger.info(f"prefer scoring count: {len(query_list)}")
        score_list = []
        count = 0
        for i in range(0, len(query_list), batch_size):
            chunk_query_list = query_list[i:i + batch_size]
            chunk_doc_list = doc_list[i:i + batch_size]
            llm_scores = llm_preferred(self.llm, chunk_query_list, chunk_doc_list, prompt)
            score_list.extend(llm_scores)
            qd_pair_scores = [{'query': self._encrypt(query), 'corpus': self._encrypt(doc), 'score': score}
                              for query, doc, score in zip(chunk_query_list, chunk_doc_list, llm_scores)]
            write_jsonl_to_file(qd_pair_scores, llm_scores_path, 'a')
            logger.info(f"The {count + 1}st LLM scoring success")
            count += 1

        return score_list

    def _generate_qd_pairs(self,
                           split_doc_list: list[str],
                           question_number: int,
                           origin_train_data_path: str,
                           prompt: str,
                           batch_size: int):
        logger.info(f"query document pair generation {len(split_doc_list)}")
        query_list = []
        doc_list = []
        count = 0
        for i in range(0, len(split_doc_list), batch_size):
            chunk_doc_list = split_doc_list[i:i + batch_size]  # 切片获取当前块的数据
            doc_queries = generate_qa_embedding_pairs(self.llm, chunk_doc_list, prompt, question_number)
            qd_pairs = []
            for doc, queries in doc_queries.items():
                query_list.extend(queries)
                docs = [doc] * len(queries)
                doc_list.extend(docs)
                for query, pos_doc in zip(queries, docs):
                    qd_pairs.append({"query": self._encrypt(query), "corpus": self._encrypt(pos_doc)})
            # 按块写文件
            write_jsonl_to_file(qd_pairs, origin_train_data_path, 'a')
            logger.info(f"The {count + 1}st query document pair generated success")
            count += 1

        return query_list, doc_list

    def _encrypt(self, text):
        if self.encrypt_fn is not None:
            result = self.encrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= STR_MAX_LEN:
                return result
            else:
                raise ValueError(f"callback function {self.encrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {STR_MAX_LEN}.")
        else:
            return text

    def _decrypt(self, text):
        if self.decrypt_fn is not None:
            result = self.decrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= STR_MAX_LEN:
                return result
            else:
                raise ValueError(f"callback function {self.decrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {STR_MAX_LEN}.")

        else:
            return text
