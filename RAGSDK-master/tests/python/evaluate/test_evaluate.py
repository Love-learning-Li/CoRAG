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
from unittest.mock import MagicMock, patch
from ragas.evaluation import EvaluationResult

from mx_rag.embedding.local import TextEmbedding
from mx_rag.evaluate import RAGEvaluator
from mx_rag.llm import Text2TextLLM


class TestRAGEvaluator(unittest.TestCase):
    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_evaluate(self, evaluate_mock):
        dataset = {
            "user_input": ["世界上最高的山峰是哪座？"],
            "response": ["珠穆朗玛峰"],
            "retrieved_contexts": [["世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8848米。"]]
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        scores = [{"answer_relevancy": 0.09, "context_utilization": 0.01, "faithfulness": 0.5}]
        result = MagicMock(spec=EvaluationResult)
        result.scores = scores
        evaluate_mock.return_value = result
        scores = evaluator.evaluate(
            metrics=["answer_relevancy", "context_utilization", "faithfulness"],
            dataset=dataset,
            language="chinese"
        )
        self.assertEqual(scores, {'answer_relevancy': [0.09], 'context_utilization': [0.01], 'faithfulness': [0.5]})

    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_evaluate_with_unsupported_metric(self, evaluate_mock):
        dataset = {
            "user_input": ["What is the capital of France?"],
            "response": ["Paris"],
            "retrieved_contexts": [["Paris is the capital of France."]]
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        scores = [{"faithfulness": 1.0}]
        result = MagicMock(spec=EvaluationResult)
        result.scores = scores
        evaluate_mock.return_value = result
        # Only 'faithfulness' is supported, 'unsupported_metric' should be filtered out
        output = evaluator.evaluate(
            metrics=["faithfulness", "unsupported_metric", "faithfulness"],
            dataset=dataset,
            language="english"
        )
        self.assertEqual(output, {'faithfulness': [1.0]})

    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_evaluate_with_empty_metrics(self, evaluate_mock):
        dataset = {
            "user_input": ["What is the capital of France?"],
            "response": ["Paris"],
            "retrieved_contexts": [["Paris is the capital of France."]]
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                metrics=[],
                dataset=dataset,
                language="english"
            )

    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_evaluate_handles_ragas_value_error(self, evaluate_mock):
        dataset = {
            "user_input": ["What is the capital of France?"],
            "response": ["Paris"],
            "retrieved_contexts": [["Paris is the capital of France."]]
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        evaluate_mock.side_effect = ValueError("ragas error")
        result = evaluator.evaluate(
            metrics=["faithfulness"],
            dataset=dataset,
            language="english"
        )
        self.assertIsNone(result)

    @patch("mx_rag.evaluate.rag_evaluator.evaluate", autospec=True)
    def test_evaluate_handles_ragas_general_exception(self, evaluate_mock):
        dataset = {
            "user_input": ["What is the capital of France?"],
            "response": ["Paris"],
            "retrieved_contexts": [["Paris is the capital of France."]]
        }
        llm = MagicMock(spec=Text2TextLLM)
        embeddings = MagicMock(spec=TextEmbedding)
        evaluator = RAGEvaluator(llm=llm, embeddings=embeddings)
        evaluate_mock.side_effect = Exception("unexpected error")
        result = evaluator.evaluate(
            metrics=["faithfulness"],
            dataset=dataset,
            language="english"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
