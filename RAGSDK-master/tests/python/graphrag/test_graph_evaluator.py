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

import numpy as np
import unittest
from unittest.mock import Mock, patch
from mx_rag.graphrag.graph_evaluator import GraphEvaluator
from mx_rag.llm import Text2TextLLM, LLMParameterConfig


class TestGraphEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up a GraphEvaluator instance with mocked dependencies."""
        self.mock_llm = Mock(spec = Text2TextLLM)
        self.mock_llm_config = Mock(spec = LLMParameterConfig)
        self.evaluator = GraphEvaluator(self.mock_llm, self.mock_llm_config)

    def test_calculate(self):
        """Test the calculate method for precision, recall, and F1 score."""
        self.assertEqual(self.evaluator._calculate(10, 2, 3), [0.8, 0.7273, 0.7619])
        self.assertEqual(self.evaluator._calculate(0, 0, 0), [np.nan, np.nan, np.nan])
        self.assertEqual(self.evaluator._calculate(5, 5, 2), [np.nan, np.nan, np.nan])

    def test_safe_len(self):
        """Test the _safe_len method for handling objects with/without len."""
        self.assertEqual(self.evaluator._safe_len([1, 2, 3]), 3)
        self.assertEqual(self.evaluator._safe_len(None), 0)
        self.assertEqual(self.evaluator._safe_len("test"), 4)

    def test_remove_empty_lines(self):
        """Test the _remove_empty_lines method for removing empty lines."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        self.assertEqual(self.evaluator._remove_empty_lines(text), "Line 1\nLine 2\nLine 3")

    def test_extract_entities_from_text(self):
        """Test the _extract_entities_from_text method for extracting entities."""
        text = "Some text with entities [entity1', 'entity2', 'entity3'] in it."
        self.assertEqual(
            self.evaluator._extract_entities_from_text(text),
            ["entity1", "entity2", "entity3"]
        )
        self.assertEqual(self.evaluator._extract_entities_from_text("No entities here."), [])

    def test_count_origin(self):
        """Test the count_origin method for counting original entities/relations."""
        entity_relations = [{"Head": "A", "Relation": "B", "Tail": "C"}]
        event_entity_relations = [{"Entity": ["E1", "E2"]}]
        event_relations = [{"Head": "X", "Relation": "Y", "Tail": "Z"}]
        self.assertEqual(self.evaluator._count_origin(entity_relations,
                                                      event_entity_relations, event_relations), [1, 2, 1])

    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._remove_empty_lines")
    def test_count_more(self, mock_remove_empty_lines):
        """Test the count_more method for counting unrecognized entities/relations."""
        mock_remove_empty_lines.side_effect = lambda x: x
        task1 = "Triple1\nTriple2"
        task2 = "{'Event': 'E1', 'Entity': ['E2', 'E3']}\n{'Event': 'E4', 'Entity': ['E5']}"
        task3 = "All recognized"
        self.assertEqual(self.evaluator._count_more(task1, task2, task3), [2, 3, 0])

    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._remove_empty_lines")
    def test_count_incorrect(self, mock_remove_empty_lines):
        """Test the count_incorrect method for counting incorrect entities/relations."""
        mock_remove_empty_lines.side_effect = lambda x: x
        task1 = "Triple1\nTriple2"
        task2 = "{'Event': 'E1', 'Entity': ['E2', 'E3']}\n{'Event': 'E4', 'Entity': ['E5']}"
        task3 = "all correct"
        self.assertEqual(self.evaluator._count_incorrect(task1, task2, task3), [2, 3, 0])

    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._get_more")
    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._get_incorrect")
    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._count_origin")
    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._count_more")
    @patch("mx_rag.graphrag.graph_evaluator.GraphEvaluator._count_incorrect")
    def test_evaluate(
        self, mock_count_incorrect, mock_count_more, mock_count_origin, mock_get_incorrect, mock_get_more):
        """Test the evaluate method for aggregating results."""
        mock_count_origin.return_value = [10, 5, 8]
        mock_get_more.return_value = ["more1", "more2", "more3"]
        mock_count_more.return_value = [2, 1, 3]
        mock_get_incorrect.return_value = ["incorrect1", "incorrect2", "incorrect3"]
        mock_count_incorrect.return_value = [1, 0, 2]

        relations = [{"raw_text": "text", "entity_relations": [], "event_entity_relations": [], "event_relations": []}]
        with patch("mx_rag.graphrag.graph_evaluator.logger.info") as mock_logger:
            self.evaluator.evaluate(relations)
            mock_logger.assert_called()  # Ensure logging is called
        relations = [{"raw_text": "text"*1025, "entity_relations": [], "event_entity_relations": [], "event_relations": []}]
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(relations)

