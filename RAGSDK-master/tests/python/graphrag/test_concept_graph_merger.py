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
from unittest.mock import Mock, patch
from collections import Counter
from mx_rag.graphrag.concept_graph_merger import ConceptGraphMerger


class TestConceptGraphMerger(unittest.TestCase):
    def setUp(self):
        """Set up a mock GraphStore and ConceptGraphMerger instance."""
        self.mock_graph = Mock()
        self.merger = ConceptGraphMerger(self.mock_graph)

    def test_init(self):
        """Test initialization of ConceptGraphMerger."""
        self.assertIsInstance(self.merger.graph, Mock)
        self.assertEqual(self.merger.entity_concepts, {})
        self.assertEqual(self.merger.relation_concepts, {})
        self.assertEqual(self.merger.event_concepts, {})
        self.assertEqual(self.merger.concept_counter, Counter())
        self.assertEqual(self.merger.node_type_counter, {"entity": 0, "relation": 0, "event": 0})

    def test_parse_concept_string(self):
        """Test parsing of concept strings."""
        result = ConceptGraphMerger.parse_concept_string("concept1, concept2, concept1")
        self.assertEqual(result, ["concept1", "concept2"])

        result = ConceptGraphMerger.parse_concept_string("  concept1 , concept2  ")
        self.assertEqual(result, ["concept1", "concept2"])

        result = ConceptGraphMerger.parse_concept_string("")
        self.assertEqual(result, [])

    @patch.object(ConceptGraphMerger, "_process_concept_data")
    @patch.object(ConceptGraphMerger, "_process_synset")
    @patch.object(ConceptGraphMerger, "_update_graph_attributes")
    def test_merge_concepts_and_synset(self, mock_update, mock_synset, mock_concept_data):
        """Test merging concepts and synsets."""
        concept_data = [{"node": "node1", "conceptualized_node": "concept1", "node_type": "entity"}]
        synset_list = [["concept1", "concept2"]]

        self.merger.merge_concepts_and_synset(concept_data, synset_list)

        mock_concept_data.assert_called_once_with(concept_data)
        mock_synset.assert_called_once_with(synset_list)
        mock_update.assert_called_once()

    def test_process_concept_data(self):
        """Test processing of concept data."""
        concept_data = [
            {"node": "node1", "conceptualized_node": "concept1, concept2", "node_type": "entity"},
            {"node": "node2", "conceptualized_node": "concept3", "node_type": "relation"},
        ]
        self.merger._process_concept_data(concept_data)

        self.assertEqual(self.merger.entity_concepts, {"node1": ["concept1", "concept2"]})
        self.assertEqual(self.merger.relation_concepts, {"node2": ["concept3"]})
        self.assertEqual(self.merger.node_type_counter, {"entity": 1, "relation": 1, "event": 0})

    def test_process_synset(self):
        """Test processing of synsets."""
        self.merger.concept_counter = Counter({"concept1": 3, "concept2": 1})
        synset_list = [["concept1", "concept2"]]

        self.merger._process_synset(synset_list)

        self.assertEqual(self.merger.concept_to_synset, {"concept1": [
                         "concept1", "concept2"], "concept2": ["concept1", "concept2"]})
        self.assertIn(("concept1", "concept2"), self.merger.synset_counter)

    @patch.object(ConceptGraphMerger, "_process_node_if_not_seen")
    def test_update_graph_attributes(self, mock_process_node):
        """Test updating graph attributes."""
        self.mock_graph.get_edges.return_value = [("node1", "node2", {"relation": "rel1"})]
        self.merger.relation_concepts = {"rel1": ["concept1"]}
        self.merger.concept_to_synset = {"concept1": ["synset1"]}

        self.merger._update_graph_attributes()

        mock_process_node.assert_any_call("node1", set(), [])
        mock_process_node.assert_any_call("node2", set(), [])
        self.mock_graph.update_node_attributes_batch.assert_called_once()
        self.mock_graph.update_edge_attributes_batch.assert_called_once()

    def test_process_node_if_not_seen(self):
        """Test processing a node if not already processed."""
        self.mock_graph.get_node_attributes.return_value = "entity"
        self.merger.entity_concepts = {"node1": ["concept1"]}
        self.merger.concept_to_synset = {"concept1": ["synset1"]}

        processed_nodes = set()
        node_updates = []

        self.merger._process_node_if_not_seen("node1", processed_nodes, node_updates)

        self.assertIn("node1", processed_nodes)
        self.assertEqual(node_updates, [("node1", {"concepts": "concept1", "synset": "synset1"})])
