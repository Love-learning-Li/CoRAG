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

from mx_rag.graphrag.graph_conceptualizer import (
    GraphConceptualizer,
    extract_event_nodes,
    extract_entity_nodes,
    extract_relation_edges,
    _check_conceptualizer_prompts,
)
from mx_rag.utils.common import Lang
from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.graphrag.graphs.graph_store import GraphStore


class TestGraphConceptualizer(unittest.TestCase):
    def setUp(self):
        """Set up mocks for GraphConceptualizer dependencies."""
        self.mock_llm = Mock(spec=Text2TextLLM)
        self.mock_llm.model_name = "mock_model"
        self.mock_llm.chat.return_value = "mock_response"
        self.mock_graph = Mock(spec=GraphStore)
        self.mock_graph.get_nodes_by_attribute.return_value = ["node1", "node2"]
        self.mock_graph.get_edge_attribute_values.return_value = [
            "relation1",
            "relation2",
        ]

    def test_initialization(self):
        """Test initialization of GraphConceptualizer."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            sample_num=None,
            lang=Lang.EN,
        )
        self.assertEqual(conceptualizer.llm, self.mock_llm)
        self.assertEqual(conceptualizer.graph, self.mock_graph)
        self.assertIsNone(conceptualizer.sample_num)
        self.assertEqual(conceptualizer.events, ["node1", "node2"])
        self.assertEqual(conceptualizer.entities, ["node1", "node2"])
        self.assertEqual(conceptualizer.relations, ["relation1", "relation2"])

    def test_initialization_with_chinese_language(self):
        """Test initialization with Chinese language setting."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.CH,
        )
        # Should use Chinese prompts
        self.assertIn("事件", conceptualizer.prompts["event"])
        self.assertIn("实体", conceptualizer.prompts["entity"])
        self.assertIn("关系", conceptualizer.prompts["relation"])

    def test_initialization_with_custom_prompts(self):
        """Test initialization with custom prompts."""
        custom_entity_prompt = "Custom entity prompt [ENTITY]"
        custom_event_prompt = "Custom event prompt [EVENT]"
        custom_relation_prompt = "Custom relation prompt [RELATION]"

        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            prompts={
                "event": custom_event_prompt,
                "entity": custom_entity_prompt,
                "relation": custom_relation_prompt,
            },
        )

        self.assertEqual(conceptualizer.prompts["entity"], custom_entity_prompt)
        self.assertEqual(conceptualizer.prompts["event"], custom_event_prompt)
        self.assertEqual(conceptualizer.prompts["relation"], custom_relation_prompt)

    def test_conceptualize(self):
        """Test the conceptualize method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            sample_num=1,
            lang=Lang.EN,
        )
        with (
            patch.object(
                conceptualizer,
                "_conceptualize_event",
                return_value={"node_type": "event"},
            ),
            patch.object(
                conceptualizer,
                "_conceptualize_entity",
                return_value={"node_type": "entity"},
            ),
            patch.object(
                conceptualizer,
                "_conceptualize_relation",
                return_value={"node_type": "relation"},
            ),
        ):
            result = conceptualizer.conceptualize()
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["node_type"], "event")
            self.assertEqual(result[1]["node_type"], "entity")
            self.assertEqual(result[2]["node_type"], "relation")

    def test_conceptualize_empty_graph(self):
        """Test conceptualize with empty graph."""
        self.mock_graph.get_nodes_by_attribute.return_value = []
        self.mock_graph.get_edge_attribute_values.return_value = []

        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
        )

        result = conceptualizer.conceptualize()
        self.assertEqual(result, [])

    def test_conceptualize_event(self):
        """Test the _conceptualize_event method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        event = "event1"
        conceptualizer.prompts["event"] = "Event: [EVENT]"
        result = conceptualizer._conceptualize_event(event)
        self.mock_llm.chat.assert_called_once_with("Event: event1")
        self.assertEqual(result["node"], event)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "event")

    def test_conceptualize_entity(self):
        """Test the _conceptualize_entity method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        entity = "entity1"
        self.mock_graph.predecessors.return_value = ["pred1"]
        self.mock_graph.successors.return_value = ["succ1"]
        self.mock_graph.get_edge_attributes.side_effect = (
            lambda src, tgt, attr: f"{src} -> {tgt}"
        )
        conceptualizer.prompts["entity"] = "Entity: [ENTITY] Context: [CONTEXT]"
        result = conceptualizer._conceptualize_entity(entity)
        self.mock_llm.chat.assert_called_once()
        self.assertIn("pred1 -> entity1", self.mock_llm.chat.call_args[0][0])
        self.assertIn("entity1 -> succ1", self.mock_llm.chat.call_args[0][0])
        self.assertEqual(result["node"], entity)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "entity")

    def test_conceptualize_entity_no_neighbors(self):
        """Test _conceptualize_entity with no neighboring nodes."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        entity = "isolated_entity"
        self.mock_graph.predecessors.return_value = []
        self.mock_graph.successors.return_value = []
        conceptualizer.prompts["entity"] = "Entity: [ENTITY] Context: [CONTEXT]"

        result = conceptualizer._conceptualize_entity(entity)

        # Should have empty context
        call_args = self.mock_llm.chat.call_args[0][0]
        self.assertIn("Context: ", call_args)
        self.assertEqual(result["node"], entity)

    def test_conceptualize_relation(self):
        """Test the _conceptualize_relation method."""
        conceptualizer = GraphConceptualizer(
            llm=self.mock_llm,
            graph=self.mock_graph,
            lang=Lang.EN,
        )
        relation = "relation1"
        conceptualizer.prompts["relation"] = "Relation: [RELATION]"
        result = conceptualizer._conceptualize_relation(relation)
        self.mock_llm.chat.assert_called_once_with("Relation: relation1")
        self.assertEqual(result["node"], relation)
        self.assertEqual(result["conceptualized_node"], "mock_response")
        self.assertEqual(result["node_type"], "relation")

    def test_extract_event_nodes(self):
        """Test extract_event_nodes utility function."""
        mock_graph = Mock()
        mock_graph.get_nodes_by_attribute.return_value = ["event1", "event2"]

        result = extract_event_nodes(mock_graph)

        mock_graph.get_nodes_by_attribute.assert_called_once_with(
            key="type", value="event"
        )
        self.assertEqual(result, ["event1", "event2"])

    def test_extract_entity_nodes(self):
        """Test extract_entity_nodes utility function."""
        mock_graph = Mock()
        mock_graph.get_nodes_by_attribute.return_value = ["entity1", "entity2"]

        result = extract_entity_nodes(mock_graph)

        mock_graph.get_nodes_by_attribute.assert_called_once_with(
            key="type", value="entity"
        )
        self.assertEqual(result, ["entity1", "entity2"])

    def test_extract_relation_edges(self):
        """Test extract_relation_edges utility function."""
        mock_graph = Mock()
        mock_graph.get_edge_attribute_values.return_value = [
            "rel1",
            "rel2",
            "rel1",
        ]  # Duplicate rel1

        result = extract_relation_edges(mock_graph)

        mock_graph.get_edge_attribute_values.assert_called_once_with(key="relation")
        # Should remove duplicates
        self.assertEqual(set(result), {"rel1", "rel2"})
        self.assertEqual(len(result), 2)

    def test_check_conceptualizer_prompt_valid(self):
        """Test _check_conceptualizer_prompt with valid inputs."""
        self.assertTrue(_check_conceptualizer_prompts(None))
        self.assertTrue(
            _check_conceptualizer_prompts(
                {
                    "event": "valid prompt",
                    "entity": "valid prompt",
                    "relation": "valid prompt",
                }
            )
        )
        self.assertTrue(
            _check_conceptualizer_prompts(
                {"event": "a" * 1000, "entity": "a" * 1000, "relation": "a" * 1000}
            )
        )  # Within limit

    def test_check_conceptualizer_prompt_invalid(self):
        """Test _check_conceptualizer_prompt with invalid inputs."""
        from mx_rag.utils.common import MAX_PROMPT_LENGTH

        self.assertFalse(
            _check_conceptualizer_prompts({"event": "", "entity": "", "relation": ""})
        )  # Empty string
        self.assertFalse(
            _check_conceptualizer_prompts(
                {
                    "event": "a" * (MAX_PROMPT_LENGTH + 1),
                    "entity": "a" * (MAX_PROMPT_LENGTH + 1),
                    "relation": "a" * (MAX_PROMPT_LENGTH + 1),
                }
            )
        )  # Too long
        self.assertFalse(
            _check_conceptualizer_prompts(
                {"event": 123, "entity": 123, "relation": 123}
            )
        )  # Not a string
        self.assertFalse(
            _check_conceptualizer_prompts(
                {"event": ["list"], "entity": ["list"], "relation": ["list"]}
            )
        )  # Not a string

    def test_parameter_validation_invalid_llm(self):
        """Test parameter validation for invalid LLM."""
        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm="not_an_llm",
                graph=self.mock_graph,
            )

    def test_parameter_validation_invalid_graph(self):
        """Test parameter validation for invalid graph."""
        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph="not_a_graph",
            )

    def test_parameter_validation_invalid_sample_num(self):
        """Test parameter validation for invalid sample_num."""
        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                sample_num=-1,  # Negative number
            )

        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                sample_num=1000001,  # Too large
            )

    def test_parameter_validation_invalid_lang(self):
        """Test parameter validation for invalid language."""
        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                lang="invalid_lang",
            )

    def test_parameter_validation_invalid_prompts(self):
        """Test parameter validation for invalid custom prompts."""
        from mx_rag.utils.common import MAX_PROMPT_LENGTH

        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                prompts={"event": "", "entity": "", "relation": ""},  # Empty prompt
            )

        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                prompts={
                    "event": "a" * (MAX_PROMPT_LENGTH + 1),
                    "entity": "a" * (MAX_PROMPT_LENGTH + 1),
                    "relation": "a" * (MAX_PROMPT_LENGTH + 1),
                },  # Too long
            )

        with self.assertRaises(Exception):  # Should raise validation error
            GraphConceptualizer(
                llm=self.mock_llm,
                graph=self.mock_graph,
                prompts={"event": 123, "entity": 123, "relation": 123},  # Not a string
            )


if __name__ == "__main__":
    unittest.main()
