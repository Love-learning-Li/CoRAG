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
from typing import Dict
import unittest
from unittest.mock import patch
from dataclasses import dataclass
import torch

from mx_rag.embedding.local import TextEmbedding


class TestTextEmbedding(unittest.TestCase):
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
        hidden_size = 512

    class Model:
        def __init__(self, test_embed_length, config):
            self.device = 'cpu'
            self.test_embed_length = test_embed_length
            self.config = config

        def __call__(self, *args, **kwargs):
            input_ids = args[0]
            last_hidden_state = torch.rand(input_ids.shape + (self.test_embed_length,))
            return TestTextEmbedding.BatchEncoding(last_hidden_state=last_hidden_state)

        def half(self):
            return self

        def to(self, *args):
            return self

        def eval(self):
            return self

    class Pooling:
        def __int__(self):
            pass

        def forward(self, d):
            return {"sentence_embedding": d.get("token_embeddings")}

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            batch_text = args[0]
            max_length = kwargs.pop('max_length', 512)
            rand_token_len = random.randint(1, max_length)
            input_ids = torch.rand((len(batch_text), rand_token_len))
            attention_mask = torch.ones((len(batch_text), rand_token_len))
            return TestTextEmbedding.BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    @patch("sentence_transformers.models.Pooling")
    def test_encode_success_fp16_mean(self,
                                      pooling_mock,
                                      torch_avail_mock,
                                      tok_pre_mock,
                                      model_pre_mock,
                                      dir_check_mock):
        pooling_mock.return_value = self.Pooling()
        model_pre_mock.return_value = self.Model(1024, self.PretrainedConfig())
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = False

        embed = TextEmbedding(model_path=self.model_path,
                              pooling_method='mean')

        texts = ['test_txt'] * 100
        ret = embed.embed_documents(texts=texts)
        self.assertEqual((len(ret), len(ret[0])), (len(texts), 1024))

        texts = ['test_txt'] * 1000
        ret = embed.embed_documents(texts=texts)
        self.assertEqual((len(ret), len(ret[0])), (len(texts), 1024))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    @patch("sentence_transformers.models.Pooling")
    def test_encode_success_fp32_cls(self,
                                     pooling_mock,
                                     torch_avail_mock,
                                     tok_pre_mock,
                                     model_pre_mock,
                                     dir_check_mock):
        pooling_mock.return_value = self.Pooling()
        model_pre_mock.return_value = self.Model(1024, self.PretrainedConfig())
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = True

        embed = TextEmbedding(model_path=self.model_path,
                              use_fp16=False)

        texts = ['test_txt'] * 100
        ret = embed.embed_documents(texts=texts)
        self.assertEqual((len(ret), len(ret[0])), (len(texts), 1024))

        texts = ['test_txt'] * 1000
        ret = embed.embed_documents(texts=texts)
        self.assertEqual((len(ret), len(ret[0])), (len(texts), 1024))

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.is_torch_npu_available")
    @patch("sentence_transformers.models.Pooling")
    def test_encode_failed_invalid_pooling(self,
                                           pooling_mock,
                                           torch_avail_mock,
                                           tok_pre_mock,
                                           model_pre_mock,
                                           dir_check_mock):
        pooling_mock.return_value = self.Pooling()
        dir_check_mock.return_value = True
        model_pre_mock.return_value = self.Model(1024, self.PretrainedConfig())
        tok_pre_mock.return_value = self.Tokenizer()
        torch_avail_mock.return_value = True
        with self.assertRaises(ValueError):
            embed = TextEmbedding(model_path=self.model_path,
                                  pooling_method='no valid')
        embed = TextEmbedding(model_path=self.model_path)
        texts = ['test_txt'] * 100
        embed.embed_documents(texts)


if __name__ == '__main__':
    unittest.main()
