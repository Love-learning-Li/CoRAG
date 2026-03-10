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

import torch
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available
from sentence_transformers.models import Pooling

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, TEXT_MAX_LEN, validate_list_str, \
    BOOL_TYPE_CHECK_TIP, STR_MAX_LEN, MAX_PATH_LENGTH, MAX_BATCH_SIZE, validate_lock, GB, get_model_max_input_length
from mx_rag.utils.file_check import SecDirCheck, safetensors_check

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. TextEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. TextEmbedding will run on cpu.")


class TextEmbedding(Embeddings):
    @validate_params(
        model_path=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_PATH_LENGTH,
                        message="param must be str and str length range [0, 1024]"),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        pooling_method=dict(validator=lambda x: x in ["cls", "mean", 'max', "lasttoken"],
                            message="param must be in ['cls', 'mean', 'max', 'lasttoken']"),
        lock=dict(
            validator=lambda x: x is None or validate_lock(x),
            message="param must be one of None, multiprocessing.Lock(), threading.Lock()")
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 use_fp16: bool = True,
                 pooling_method: str = 'cls',
                 lock=None):
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        self.pooling_method = pooling_method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.pooling = Pooling(self.model.config.hidden_size, pooling_mode=self.pooling_method)

        self.model_lock = lock

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

        return TextEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                           "str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param must be int and value range [1, {MAX_BATCH_SIZE}]")
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32) -> List[List[float]]:
        result = []

        max_input_length = get_model_max_input_length(self.model.config)
        if max_input_length == 0:
            raise ValueError("get model max input length failed")

        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]

            encode_texts = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_input_length, return_tensors='pt').to(
                self.model.device)

            attention_mask = encode_texts.attention_mask
            model_output = self._safe_call_model(encode_texts, attention_mask)
            last_hidden_state = model_output.last_hidden_state
            embeddings = self.pooling.forward({"token_embeddings": last_hidden_state,
                                               "attention_mask": attention_mask})["sentence_embedding"]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu().tolist()
            result.extend(embeddings)

        return result

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= STR_MAX_LEN,
                  message="param must be str and value range [1, 128 * 1024 * 1024]")
    )
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    def _safe_call_model(self, encode_texts, attention_mask):
        def _call_model():
            with torch.no_grad():
                model_output = self.model(encode_texts.input_ids, attention_mask, return_dict=True)
            return model_output

        if self.model_lock is not None:
            with self.model_lock:
                return _call_model()
        else:
            return _call_model()
