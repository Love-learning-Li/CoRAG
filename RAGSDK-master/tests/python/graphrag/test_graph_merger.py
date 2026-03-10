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
from mx_rag.graphrag.graph_merger import merge_relations_into_graph
from mx_rag.utils.common import Lang


class TestMergeRelationsIntoGraph(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_graph = Mock()
        self.mock_graph.number_of_nodes.return_value = 10
        self.mock_graph.number_of_edges.return_value = 20
        self.mock_graph.density.return_value = 0.1
        self.mock_logger = patch("mx_rag.graphrag.graph_merger.logger").start()
        self.mock_tqdm = patch("mx_rag.graphrag.graph_merger.tqdm", side_effect=lambda x, **kwargs: x).start()

    def tearDown(self):
        """Clean up after each test method."""
        patch.stopall()

    def test_merge_with_valid_relations(self):
        """Test merging valid relations into the graph."""
        relations = [
            {
                "raw_text": "Sample text",
                "file_id": "123",
                "entity_relations": [
                    {"Head": "Entity1", "Relation": "related_to", "Tail": "Entity2"}
                ],
                "event_entity_relations": [
                    {"Event": "Event1", "Entity": ["Entity1", "Entity2"]}
                ],
                "event_relations": [
                    {"Head": "Event1", "Relation": "causes", "Tail": "Event2"}
                ],
            }
        ]

        merge_relations_into_graph(self.mock_graph, relations, language=Lang.EN)

        # Verify graph operations
        self.mock_graph.add_node.assert_called_once_with("Sample text")
        self.mock_graph.update_node_attribute.assert_any_call("Sample text", "type", "raw_text")
        self.mock_graph.update_node_attribute.assert_any_call("Sample text", "file_id", "123", append=True)
        self.mock_graph.add_edge.assert_any_call("Entity1", "Entity2", relation="related_to")
        self.mock_graph.add_edge.assert_any_call("Entity1", "Event1", relation="participate")
        self.mock_graph.add_edge.assert_any_call("Event1", "Event2", relation="causes")

    def test_merge_with_invalid_relations(self):
        """Test merging with invalid relation formats."""
        relations = [
            {"raw_text": "Sample text", "file_id": "123", "entity_relations": ["invalid_format"]}
        ]

        merge_relations_into_graph(self.mock_graph, relations, language=Lang.EN)

        # Verify warnings are logged
        self.mock_logger.warning.assert_called_with("Wrong relation")

    def test_merge_with_missing_raw_text(self):
        """Test merging when raw_text is missing."""
        relations = [{"file_id": "123"}]

        merge_relations_into_graph(self.mock_graph, relations, language=Lang.EN)

        # Verify warnings are logged
        self.mock_logger.warning.assert_called_with("Missing raw_text in relation")

    def test_merge_creates_indices(self):
        """Test that graph indices are created."""
        self.mock_graph.create_index_for_edge = Mock()
        self.mock_graph.create_index_for_node = Mock()

        merge_relations_into_graph(self.mock_graph, [], language=Lang.EN)

        # Verify index creation
        self.mock_graph.create_index_for_edge.assert_called_once()
        self.mock_graph.create_index_for_node.assert_called_once()

    def test_merge_logs_graph_stats(self):
        """Test that graph statistics are logged."""
        merge_relations_into_graph(self.mock_graph, [], language=Lang.EN)

        # Verify stats are logged
        self.mock_logger.info.assert_any_call("Creating indices for graph...")
        self.mock_logger.info.assert_any_call(
            "Graph stats - Nodes: 10, Edges: 20, Density: 0.100000"
        )
