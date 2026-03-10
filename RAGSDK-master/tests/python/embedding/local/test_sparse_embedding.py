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
from mx_rag.embedding.local import SparseEmbedding


class TestSparseEmbedding(unittest.TestCase):
    model_path = "/model/embedding"

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
        hidden_size = 1024

    class Model:
        def __init__(self, test_embed_length, config):
            self.device = 'cpu'
            self.test_embed_length = test_embed_length
            self.config = config

        def __call__(self, *args, **kwargs):
            input_ids = kwargs.get('input_ids')
            last_hidden_state = torch.rand(input_ids.shape + (self.test_embed_length,), dtype=torch.float32)
            return TestSparseEmbedding.BatchEncoding(last_hidden_state=last_hidden_state)

        def half(self):
            return self

        def to(self, *args):
            return self

        def eval(self):
            return self

    class Config:
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            batch_text = args[0]
            max_length = kwargs.pop('max_length', 512)
            rand_token_len = random.randint(1, max_length)
            input_ids = torch.rand((len(batch_text), rand_token_len))
            attention_mask = torch.ones((len(batch_text), rand_token_len))
            return TestSparseEmbedding.BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

        def cls_token_id(self):
            return 0

        def eos_token_id(self):
            return 1

        def pad_token_id(self):
            return 2

        def unk_token_id(self):
            return 3

    @patch("torch.load")
    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    def test_encode_success(self,
                            torch_avail_mock,
                            tok_pre_mock,
                            model_pre_mock,
                            dir_check_mock,
                            torch_load_mock):
        model_pre_mock.return_value = self.Model(1024, self.PretrainedConfig())
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = False
        dir_check_mock.return_value = True
        torch_load_mock.return_value = {"weight": torch.randn(1, 1024).to(torch.float32),
                                        "bias": torch.randn(1).to(torch.float32)}

        embed = SparseEmbedding(model_path=self.model_path)

        text = 'test_txt'
        ret = embed.embed_query(text=text)
        self.assertEqual(type(ret), dict)

        texts = ['test_txt'] * 100
        ret = embed.embed_documents(texts=texts)
        self.assertEqual(len(ret), len(texts))

        embed = SparseEmbedding.create(model_path=self.model_path)
        assert isinstance(embed, SparseEmbedding)

        embed = SparseEmbedding.create()
        self.assertEqual(embed, None)


if __name__ == '__main__':
    unittest.main()
