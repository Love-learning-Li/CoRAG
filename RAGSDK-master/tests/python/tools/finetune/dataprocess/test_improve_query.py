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


import unittest
from unittest.mock import patch

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess.improve_query import RuleComplexInstructionRewriter, improve_query


class TestImproveQuery(unittest.TestCase):

    def test_run_success(self):
        rewriter = RuleComplexInstructionRewriter()
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令子任务"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加回答的限制"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加领域知识"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令格式"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令要求"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "更改指令语言风格"), "")

    def test_run_failed(self):
        rewriter = RuleComplexInstructionRewriter()
        self.assertEqual(rewriter.get_rewrite_prompts("请问你是谁", "不存在的指令"), "")

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_type_error(self, chat):
        chat.side_effect = TypeError("Invalid argument type")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        old_query_list = ["This is a test question"]
        new_query_list = improve_query(llm, old_query_list)

        self.assertEqual(new_query_list, [''])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_timeout_error(self, chat):
        chat.side_effect = TimeoutError("Request timed out")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        old_query_list = ["This is a test question"]
        new_query_list = improve_query(llm, old_query_list)

        self.assertEqual(new_query_list, [''])

    @patch("mx_rag.llm.Text2TextLLM.chat")
    def test_make_request_generic_exception(self, chat):
        chat.side_effect = Exception("Unknown error occurred")
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", timeout=120)
        old_query_list = ["This is a test question"]
        new_query_list = improve_query(llm, old_query_list)

        self.assertEqual(new_query_list, [''])


if __name__ == '__main__':
    unittest.main()
