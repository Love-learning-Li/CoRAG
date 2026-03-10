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
from mx_rag.tools.finetune.dataprocess.generate_qd import GENERATE_QD_PROMPT
from mx_rag.tools.finetune.generator.common import BaseGenerator
from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.file_operate import write_jsonl_to_file
from mx_rag.utils.common import (validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN,
                                 MAX_PATH_LENGTH, MAX_PROMPT_LENGTH)


class EvalDataGenerator(BaseGenerator):
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        dataset_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PATH_LENGTH,
                          message=f"param must be str and str length range (0, {MAX_PATH_LENGTH}]"),
        encrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="encrypt_fun must be None or callable function"),
        decrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="decrypt_fun must be None or callable function")
    )
    def __init__(self, llm: Text2TextLLM, dataset_path: str, encrypt_fn: Callable[[str], str] = None,
                 decrypt_fn: Callable[[str], str] = None):
        super().__init__(llm, dataset_path, encrypt_fn, decrypt_fn)

    @validate_params(
        split_doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                            message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                                    f"str length range [1, {STR_MAX_LEN}]"),
        generate_qd_prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                                message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
        question_number=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 20,
                             message="param must meets: Type is int, length range (0, 20]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1024,
                        message="param must meets: Type is int, length range (0, 1024]"),

    )
    def generate_evaluate_data(self,
                               split_doc_list: list[str],
                               generate_qd_prompt: str = GENERATE_QD_PROMPT,
                               question_number: int = 3,
                               batch_size: int = 8):
        FileCheck.dir_check(self.dataset_path)
        evaluate_data_path = os.path.join(self.dataset_path, "evaluate_data.jsonl")
        if os.path.exists(evaluate_data_path):
            logger.info("embedding evaluate data has been created.")
            return

        # 流程开始
        logger.info("step Generating rough problem documentation pairs")
        query_list, doc_list = self._generate_coarsest_qd_pairs(split_doc_list, question_number,
                                                                generate_qd_prompt, batch_size)
        logger.info("step Generated rough problem documentation pairs finished")

        evaluate_data = []
        for query, doc in zip(query_list, doc_list):
            evaluate_data.append({"query": self._encrypt(query), "corpus": self._encrypt(doc)})

        write_jsonl_to_file(evaluate_data, evaluate_data_path)

        return
