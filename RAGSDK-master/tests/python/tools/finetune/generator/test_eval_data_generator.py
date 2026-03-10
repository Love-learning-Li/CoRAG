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
import unittest
from unittest.mock import patch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader
from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.generator import EvalDataGenerator
from mx_rag.utils import ClientParam


class TestEvalDataGenerator(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.realpath(os.path.join(current_dir, "../../../../data/"))
        client_param = ClientParam(use_http=True, timeout=120)
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", client_param=client_param)
        self.eval_data_generator = EvalDataGenerator(llm, self.file_path)
        if os.path.exists(os.path.join(self.file_path, "evaluate_data.jsonl")):
            os.remove(os.path.join(self.file_path, "evaluate_data.jsonl"))
        jsonl_path = os.path.join(self.file_path, "origin_train_data.jsonl")
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    @patch("mx_rag.utils.file_check.FileCheck.check_path_is_exist_and_valid")
    @patch("mx_rag.utils.file_check.SecFileCheck.check")
    @patch("mx_rag.tools.finetune.dataprocess.generate_qd.generate_qa_embedding_pairs")
    def test_generate_eval_data(self, generate_qa_mock, path_check_mock, eval_file_check_mock, dir_check_mock):
        generate_qa_mock.return_value = {"content1": ["q1"], "content2": ["q2"]}
        self.eval_data_generator.generate_evaluate_data(["test"])

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_generate_origin_document(self, dir_check_mock):
        loader_mng = LoaderMng()

        loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])

        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                     file_types=[".docx"],
                                     splitter_params={"chunk_size": 750,
                                                      "chunk_overlap": 150,
                                                      "keep_separator": False
                                                      }
                                     )
        result = self.eval_data_generator.generate_origin_document(os.path.join(self.file_path, "files"), loader_mng)
        self.assertNotEqual([], result)
