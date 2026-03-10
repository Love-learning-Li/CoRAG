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
import random
import shutil
from dataclasses import dataclass
from typing import Dict
import unittest
from unittest.mock import patch

import torch

from mx_rag.reranker.local import LocalReranker


class TestLocalReranker(unittest.TestCase):
    model_path = "/model/reranker"

    def setUp(self) -> None:
        os.makedirs(self.model_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_path)

    class BatchEncoding(Dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                error_msg = f'Dict object has no attribute [{key}]'
                raise AttributeError(error_msg)

        def __setattr__(self, key, value):
            self[key] = value

        def to(self, *args):
            return self

    @dataclass
    class PretrainedConfig:
        model_type = "bert"
        pad_token_id = 1
        max_seq_length = 512
        max_position_embeddings = 512
        hidden_size = 512

    class Model:
        def __init__(self, config):
            self.device = 'cpu'
            self.config = config

        def __call__(self, *args, **kwargs):
            input_ids = kwargs.pop('input_ids')
            logits = torch.rand(len(input_ids), 1)
            return TestLocalReranker.BatchEncoding(logits=logits)

        def half(self):
            return self

        def to(self, *args):
            return self

        def eval(self):
            return self

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            batch_text = args[0]
            max_length = kwargs.pop('max_length', 512)
            rand_token_len = random.randint(1, max_length)
            input_ids = torch.rand((len(batch_text), rand_token_len))
            return TestLocalReranker.BatchEncoding(input_ids=input_ids)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_rerank_success_fp16(self,
                                 torch_avail_mock,
                                 tok_pre_mock,
                                 model_pre_mock,
                                 dir_check_mock):
        model_pre_mock.return_value = TestLocalReranker.Model(self.PretrainedConfig())
        tok_pre_mock.return_value = TestLocalReranker.Tokenizer()
        torch_avail_mock.return_value = True

        rerank = LocalReranker(model_path=self.model_path)
        texts = ['我是小黑', '我是小红'] * 100
        ret = rerank.rerank(query='你好', texts=texts)

        self.assertEqual(ret.shape, (len(texts),))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_rerank_success_fp32(self,
                                 torch_avail_mock,
                                 tok_pre_mock,
                                 model_pre_mock,
                                 dir_check_mock):
        model_pre_mock.return_value = TestLocalReranker.Model(self.PretrainedConfig())
        tok_pre_mock.return_value = TestLocalReranker.Tokenizer()
        torch_avail_mock.return_value = False

        rerank = LocalReranker(model_path=self.model_path,
                               use_fp16=False)
        texts = ['我是小黑', '我是小红'] * 100
        ret = rerank.rerank(query='你好', texts=texts)

        self.assertEqual(ret.shape, (len(texts),))


if __name__ == '__main__':
    unittest.main()
