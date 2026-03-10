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
from unittest.mock import patch, MagicMock

from transformers import PreTrainedTokenizerBase

from mx_rag.cache import QAGenerate, QAGenerationConfig, MarkDownParser
from mx_rag.llm import Text2TextLLM


class TestQAGenerate(unittest.TestCase):

    def test_generate_qas_length_not_equal(self):
        config = QAGenerationConfig(['title1', 'title2'], ['content1'],
                                    MagicMock(spec=PreTrainedTokenizerBase), MagicMock(spec=Text2TextLLM))
        qa_generate = QAGenerate(config)
        with self.assertRaises(ValueError):
            qa_generate.generate_qa()

    @patch("mx_rag.cache.QAGenerate.generate_qa")
    def test_generate_qas_no_qas(self, mock_generate_qas):
        config = QAGenerationConfig(['title1', 'title2'], ['content1'],
                                    MagicMock(spec=PreTrainedTokenizerBase), MagicMock(spec=Text2TextLLM))
        qa_generate = QAGenerate(config)
        mock_generate_qas.return_value = []
        result = qa_generate.generate_qa()
        self.assertEqual(result, [])

    @patch("mx_rag.cache.QAGenerate._split_html_text")
    @patch("mx_rag.cache.QAGenerate._generate_qa_from_html")
    def test_generate_qas_with_qas(self, generate_mock, split_mock):
        config = QAGenerationConfig(['title1', 'title2'], ['content1', 'content2'],
                                    MagicMock(spec=PreTrainedTokenizerBase), MagicMock(spec=Text2TextLLM))
        qa_generate = QAGenerate(config)
        generate_mock.return_value = ["q1?参考段落:answer1", "q2?参考段落:answer2"]
        split_mock.return_value = "text"
        result = qa_generate.generate_qa()
        self.assertEqual(result, {'q1?': 'answer1', 'q2?': 'answer2'})

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_markdown_parse(self, dir_check_mock):
        # 创建MarkDownParser实例
        test_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data"))
        parser = MarkDownParser(test_dir)
        # 调用parse方法
        titles, contents = parser.parse()
        # 验证结果
        self.assertIn('test.md', titles)
        self.assertIn('# Test Tile\n\nthis is a test', contents)



if __name__ == '__main__':
    unittest.main()
