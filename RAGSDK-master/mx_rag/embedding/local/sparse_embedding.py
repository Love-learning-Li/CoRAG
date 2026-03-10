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
from typing import List
import torch
import numpy as np

from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoTokenizer, is_torch_npu_available, AutoModel

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, TEXT_MAX_LEN, validate_list_str, \
    BOOL_TYPE_CHECK_TIP, STR_MAX_LEN, MAX_PATH_LENGTH, MAX_BATCH_SIZE, GB, get_model_max_input_length
from mx_rag.utils.file_check import SecDirCheck, safetensors_check

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. TextEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. TextEmbedding will run on cpu.")


class SparseEmbedding(Embeddings):
    @validate_params(
        model_path=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_PATH_LENGTH,
                        message="param must be str and str length range [0, 1024]"),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 use_fp16: bool = True):
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

        self.sparse_linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                             out_features=1).to(self.model.device)
        sparse_model_path = os.path.join(self.model_path, 'sparse_linear.pt')
        sparse_state_dict = torch.load(sparse_model_path, map_location=self.model.device, weights_only=True)
        self.sparse_linear.load_state_dict(sparse_state_dict)

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            logger.error("model_path param error. ")
            return None

        return SparseEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                           "str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param must be int and value range [1, {MAX_BATCH_SIZE}]")
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32) -> List[dict[int, float]]:

        max_input_length = get_model_max_input_length(self.model.config)
        if max_input_length == 0:
            raise ValueError("get model max input length failed")

        result = self._encode(texts, batch_size, max_input_length)
        if len(result) == 0:
            raise ValueError("embedding documents text error")

        return result

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= STR_MAX_LEN,
                  message="param must be str and value range [1, 128 * 1024 * 1024]")
    )
    def embed_query(self, text: str) -> dict[int, float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding query text failed")

        return embeddings[0]

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # conver to dict
        result = {}
        unused_tokens = {self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id}
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = int(idx)
                if idx not in result:
                    result[idx] = w
                elif w > result[idx]:
                    result[idx] = w
        return result

    def _process_batch(self, batch_texts: List[str], max_length: int):
        all_lexical_weights = []
        batch_data = self.tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=max_length,
        ).to(self.model.device)
        # 使用线性层进行稀疏向量化
        with torch.no_grad():
            last_hidden_state = self.model(**batch_data, return_dict=True).last_hidden_state
        sparse_vecs = torch.relu(self.sparse_linear(last_hidden_state))
        token_weights = sparse_vecs.squeeze(-1)
        all_lexical_weights.extend(list(map(self._process_token_weights, token_weights.detach().cpu().numpy(),
                                            batch_data['input_ids'].cpu().numpy().tolist())))
        return all_lexical_weights

    def _encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[dict]:
        result = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]
            batch_result = self._process_batch(batch_texts, max_length)
            result.extend(batch_result)
        return result
