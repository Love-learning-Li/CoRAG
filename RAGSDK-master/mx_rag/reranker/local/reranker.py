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

from typing import List

from loguru import logger
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import (validate_params, MAX_DEVICE_ID, MAX_TOP_K, TEXT_MAX_LEN,
                                 validate_list_str, BOOL_TYPE_CHECK_TIP,
                                 MAX_QUERY_LENGTH, STR_MAX_LEN, MAX_PATH_LENGTH, MAX_BATCH_SIZE, GB,
                                 get_model_max_input_length, MB)
from mx_rag.utils.file_check import SecDirCheck, safetensors_check

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. LocalReranker will run on CPU.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. LocalReranker will run on CPU.")


class LocalReranker(Reranker):

    @validate_params(
        model_path=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_PATH_LENGTH,
                        message="param must be str and str length range [0, 1024]"),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        k=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_TOP_K,
               message="param must be int and value range [1, 10000]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 k: int = 1,
                 use_fp16: bool = True):
        super(LocalReranker, self).__init__(k)
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True,
                                                                        use_safetensors=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            logger.error("model_path param error. ")
            return None

        return LocalReranker(**kwargs)

    @validate_params(
        query=dict(validator=lambda x: 1 <= len(x) <= MB,
                   message="param length range [1, 1024 * 1024]"),
        texts=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, MB]),
                   message="param must meets: Type is List[str], "
                           "list length range [1, 1000 * 1000], str length range [1, 1024 * 1024]"),
        batch_size=dict(validator=lambda x: 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param value range [1, {MAX_BATCH_SIZE}]"),
    )
    def rerank(self,
               query: str,
               texts: List[str],
               batch_size: int = 32) -> np.array:
        sentence_pairs = [[query, text] for text in texts]

        max_input_length = get_model_max_input_length(self.model.config)
        if max_input_length == 0:
            raise ValueError("get model max input length failed")

        result = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            sentence_batch = sentence_pairs[start_index:start_index + batch_size]

            encode_pairs = self.tokenizer(
                sentence_batch, padding=True, truncation=True, max_length=max_input_length, return_tensors='pt').to(
                self.model.device)

            with torch.no_grad():
                model_output = self.model(**encode_pairs, return_dict=True).logits.view(-1, ).float()

            scores = model_output.cpu().numpy().tolist()
            result = result + scores

        return np.array(result)
