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
from paddle.base import libpaddle
from mx_rag.graphrag.graph_rag_model import GraphRAGModel


class TestGraphRAGModel(unittest.TestCase):
    def setUp(self):
        """Set up mock dependencies and GraphRAGModel instance."""
        self.mock_llm = Mock()
        self.mock_llm_config = Mock()
        self.mock_embed_func = Mock()
        self.mock_graph_store = Mock()
        self.mock_vector_store = Mock()
        self.mock_vector_store_concept = Mock()
        self.mock_reranker = Mock()
        
        # Setup mock return values
        self.mock_vector_store.ntotal.return_value = 0
        self.mock_vector_store_concept.ntotal.return_value = 0
        self.mock_graph_store.get_nodes.return_value = [
            ("node1", {"type": "entity"}),
            ("node2", {"type": "raw_text"}),
            ("node3", {"type": "entity", "concepts": ["concept1", "concept2"]})
        ]
        self.mock_embed_func.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Patch the _initialize_databases method to avoid actual database building during setup
        with patch.object(GraphRAGModel, '_initialize_databases'):
            self.model = GraphRAGModel(
                llm=self.mock_llm,
                llm_config=self.mock_llm_config,
                embed_func=self.mock_embed_func,
                graph_store=self.mock_graph_store,
                vector_store=self.mock_vector_store,
                vector_store_concept=self.mock_vector_store_concept,
                reranker=self.mock_reranker,
                retrieval_top_k=5,
                subgraph_depth=2,
                use_text=True
            )

    def test_init(self):
        """Test initialization of GraphRAGModel."""
        self.assertEqual(self.model.llm, self.mock_llm)
        self.assertEqual(self.model.llm_config, self.mock_llm_config)
        self.assertEqual(self.model.embed_func, self.mock_embed_func)
        self.assertEqual(self.model.graph, self.mock_graph_store)
        self.assertEqual(self.model.vector_store, self.mock_vector_store)
        self.assertEqual(self.model.vector_store_concept, self.mock_vector_store_concept)
        self.assertEqual(self.model.reranker, self.mock_reranker)
        self.assertEqual(self.model.retrieval_top_k, 5)
        self.assertEqual(self.model.subgraph_depth, 2)
        self.assertTrue(self.model.use_text)
        self.assertIsNone(self.model.subgraph)

    def test_search_index(self):
        """Test search_index method."""
        self.model.text_nodes = ["node1", "node2"]
        self.model.node_names = ["node1", "node2", "node3"]
        self.mock_embed_func.return_value = [[0.1, 0.2, 0.3]]
        self.mock_vector_store.search.return_value = (None, [[0, 1, 2]])
        
        result = self.model.search_index("test query", 3)
        
        self.assertEqual(result, ["node1", "node2"])
        self.mock_embed_func.assert_called_with(["test query"])
        self.mock_vector_store.search.assert_called_once()

    def test_search_index_error_handling(self):
        """Test search_index error handling."""
        self.mock_embed_func.side_effect = Exception("Embedding error")
        
        with self.assertRaises(Exception):
            self.model.search_index("test query", 3)

    def test_search_index_type_error(self):
        """Test search_index handling of TypeError."""
        # Mock embed_func to raise TypeError
        self.mock_embed_func.side_effect = TypeError("Type Error")

        with self.assertRaises(TypeError):
            self.model.search_index("test query", 3)

    def test_search_index_value_error(self):
        """Test search_index handling of ValueError."""
        # Mock embed_func to raise ValueError
        self.mock_embed_func.side_effect = ValueError("Value Error")

        with self.assertRaises(ValueError):
            self.model.search_index("test query", 3)

    def test_retrieve_basic(self):
        """Test basic retrieve method."""
        self.model.node_names = ["node1", "node2", "node3"]
        self.mock_embed_func.return_value = [[0.1, 0.2, 0.3]]
        self.mock_vector_store_concept.search.return_value = (None, [])
        self.mock_vector_store.search.return_value = (None, [[0, 1]])
        
        result = self.model.retrieve("test query", 2)
        
        self.assertEqual(result, ["node1", "node2"])

    def test_retrieve_with_concept_store(self):
        """Test retrieve method with concept vector store."""
        self.model.node_names = ["node1", "node2", "node3"]
        self.mock_embed_func.return_value = [[0.1, 0.2, 0.3]]
        self.mock_vector_store.search.return_value = (None, [[0, 1]])
        self.mock_vector_store_concept.search.return_value = (None, [[1, 2]])
        
        result = self.model.retrieve("test query", 2)
        
        # Should merge and deduplicate results
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2)

    def test_retrieve_error_handling(self):
        """Test retrieve method error handling."""
        self.mock_embed_func.side_effect = Exception("Embedding error")
        
        result = self.model.retrieve("test query", 2)
        
        self.assertEqual(result, [])

    @patch('mx_rag.graphrag.graph_rag_model.isinstance')
    def test_get_contexts_for_nodes_opengauss(self, mock_isinstance):
        """Test get_contexts_for_nodes with OpenGaussGraph."""
        # Mock isinstance to return True for OpenGaussGraph check
        mock_isinstance.return_value = True
        
        # Create a mock instance that mimics OpenGaussGraph behavior
        mock_graph_instance = Mock()
        mock_graph_instance.subgraph.return_value = [("u1", "rel1", "v1"), ("u2", "rel2", "v2")]
        self.model.graph = mock_graph_instance
        self.model.use_text = False
        
        result = self.model.get_contexts_for_nodes(["node1", "node2"], 1)
        
        self.assertEqual(result, ["u1 rel1 v1", "u2 rel2 v2"])

    def test_get_contexts_for_nodes_with_text(self):
        """Test get_contexts_for_nodes with text extraction."""
        self.model.use_text = True
        with patch.object(self.model, '_build_neighbor_subgraph'), \
             patch.object(self.model, '_extract_edges_with_attributes') as mock_extract:
            mock_extract.return_value = [
                ("u1", "text_conclude", "text1"),
                ("u2", "other_rel", "v2"),
                ("u3", "text_conclude", "text2")
            ]
            
            result = self.model.get_contexts_for_nodes(["node1"], 1)
            
            self.assertEqual(result, ["text1", "text2"])

    def test_reset_subgraph(self):
        """Test reset_subgraph method."""
        self.model.subgraph = Mock()
        
        self.model.reset_subgraph()
        
        self.assertIsNone(self.model.subgraph)

    @patch.object(GraphRAGModel, '_generate_answers_batch')
    @patch.object(GraphRAGModel, '_prepare_prompts_batch')
    @patch.object(GraphRAGModel, '_retrieve_nodes_batch')
    @patch.object(GraphRAGModel, '_extract_entities_batch')
    def test_generate(self, mock_extract, mock_retrieve, mock_prepare, mock_generate_answers):
        """Test generate method orchestration."""
        questions = ["What is AI?", "How does ML work?"]
        mock_extract.return_value = [["AI"], ["ML"]]
        mock_retrieve.return_value = {"AI": ["node1"], "ML": ["node2"]}
        mock_prepare.return_value = [["prompt1", "prompt2"], []]
        mock_generate_answers.return_value = ["answer1", "answer2"]
        
        result = self.model.generate(questions, max_triples=100, retrieve_only=False)
        
        self.assertEqual(result, ["answer1", "answer2"])
        mock_extract.assert_called_once_with(questions)
        mock_retrieve.assert_called_once_with([["AI"], ["ML"]])
        mock_prepare.assert_called_once_with(questions, [["AI"], ["ML"]], {"AI": ["node1"], "ML": ["node2"]}, 100)
        mock_generate_answers.assert_called_once_with(["prompt1", "prompt2"])

    def test_extract_entities_batch(self):
        """Test _extract_entities_batch method."""
        questions = ["What is AI?", "How does ML work?"]
        with patch.object(self.model, '_extract_entities_from_question') as mock_extract:
            mock_extract.side_effect = [["AI"], ["ML"]]
            
            result = self.model._extract_entities_batch(questions)
            
            self.assertEqual(result, [["AI"], ["ML"]])
            self.assertEqual(mock_extract.call_count, 2)

    def test_retrieve_nodes_batch(self):
        """Test _retrieve_nodes_batch method."""
        entities_list = [["AI", "tech"], ["ML"]]
        with patch.object(self.model, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = lambda entity, top_k: [f"node_{entity}"]
            
            result = self.model._retrieve_nodes_batch(entities_list)
            
            expected = {"AI": ["node_AI"], "tech": ["node_tech"], "ML": ["node_ML"]}
            self.assertEqual(result, expected)

    def test_gather_nodes_for_question(self):
        """Test _gather_nodes_for_question method."""
        entities = ["AI", "ML"]
        entity_to_nodes = {"AI": ["node1", "node2"], "ML": ["node2", "node3"]}
        
        result = self.model._gather_nodes_for_question(entities, entity_to_nodes)
        
        # Should deduplicate while preserving order
        self.assertEqual(result, ["node1", "node2", "node3"])

    @patch.object(GraphRAGModel, '_rerank')
    @patch.object(GraphRAGModel, 'get_contexts_for_nodes')
    def test_get_and_rerank_contexts(self, mock_get_contexts, mock_rerank):
        """Test _get_and_rerank_contexts method."""
        mock_get_contexts.return_value = ["context1", "context2", "context3"]
        mock_rerank.return_value = ["context1", "context3"]
        
        result = self.model._get_and_rerank_contexts(["node1"], "query", 2)
        
        self.assertEqual(result, ["context1", "context3"])
        mock_get_contexts.assert_called_once_with(["node1"], self.model.subgraph_depth)
        mock_rerank.assert_called_once_with(["context1", "context2"], "query")

    def test_call_llm_with_retry_success(self):
        """Test _call_llm_with_retry with successful response."""
        self.mock_llm.chat.return_value = "Success response"
        
        result = self.model._call_llm_with_retry("test prompt")
        
        self.assertEqual(result, "Success response")
        self.mock_llm.chat.assert_called_once()

    def test_call_llm_with_retry_failure(self):
        """Test _call_llm_with_retry with repeated failures."""
        self.mock_llm.chat.return_value = ""
        
        result = self.model._call_llm_with_retry("test prompt", max_retries=2)
        
        self.assertEqual(result, "")
        self.assertEqual(self.mock_llm.chat.call_count, 2)

    def test_build_node_database(self):
        """Test _build_node_database method."""
        self.mock_graph_store.get_nodes.side_effect = [
            ["node1", "node2"],  # First call for node names
            [("node1", {"type": "entity"}), ("node2", {"type": "raw_text"})]  # Second call for text nodes
        ]
        self.mock_vector_store.ntotal.return_value = 0  # Force rebuild
        self.mock_embed_func.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        self.model._build_node_database()
        
        self.assertEqual(self.model.node_names, ["node1", "node2"])
        self.assertEqual(self.model.text_nodes, ["node2"])
        self.mock_vector_store.clear.assert_called_once()
        self.mock_vector_store.add.assert_called_once()
        self.mock_vector_store.save.assert_called_once()

    def test_build_concept_database(self):
        """Test _build_concept_database method."""
        self.mock_graph_store.get_nodes.return_value = [
            ("node1", {"concepts": ["concept1", "concept2"]}),
            ("node2", {"concepts": "concept3"}),
            ("node3", {})
        ]
        self.mock_vector_store_concept.ntotal.return_value = 0  # Force rebuild
        self.mock_embed_func.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        self.model._build_concept_database()
        
        self.assertCountEqual(self.model.concepts, ["concept1", "concept2", "concept3"])
        self.mock_vector_store_concept.clear.assert_called_once()
        self.mock_vector_store_concept.add.assert_called_once()
        self.mock_vector_store_concept.save.assert_called_once()

    def test_rerank_with_reranker(self):
        """Test _rerank method with reranker."""
        self.model.use_text = True
        text_nodes = ["text1", "text2", "text3"]
        self.mock_reranker.rerank.return_value = [0.9, 0.5, 0.8]
        self.mock_reranker.rerank_top_k.return_value = ["text1", "text3"]
        
        result = self.model._rerank(text_nodes, "query")
        
        self.assertEqual(result, ["text1", "text3"])
        self.mock_reranker.rerank.assert_called_once_with("query", text_nodes)

    def test_rerank_without_reranker(self):
        """Test _rerank method without reranker using similarity."""
        self.model.use_text = True
        self.model.reranker = None
        text_nodes = ["text1", "text2"]
        self.mock_embed_func.side_effect = [
            [[0.1, 0.2, 0.3]],  # Query embedding
            [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]  # Text embeddings
        ]
        
        result = self.model._rerank(text_nodes, "query")
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), len(text_nodes))

    def test_rerank_without_text(self):
        """Test _rerank method when use_text is False."""
        self.model.use_text = False
        text_nodes = ["text1", "text1", "text2"]
        
        result = self.model._rerank(text_nodes, "query")
        
        # Should return most common items
        self.assertIn("text1", result)
        self.assertIsInstance(result, list)

    def test_add_neighbors_to_subgraph(self):
        """Test _add_neighbors_to_subgraph method."""
        self.model.subgraph = Mock()
        self.mock_graph_store.successors.return_value = ["neighbor1"]
        self.mock_graph_store.predecessors.return_value = ["pred1"]
        self.mock_graph_store.get_edge_attributes.return_value = {"relation": "knows"}
        
        visited = set()
        queue = []
        self.model._add_neighbors_to_subgraph("current_node", visited, queue, 0)
        
        self.assertIn("neighbor1", visited)
        self.assertIn("pred1", visited)
        self.assertEqual(len(queue), 2)

    def test_build_neighbor_subgraph(self):
        """Test _build_neighbor_subgraph method."""
        self.mock_graph_store.subgraph.return_value = Mock()
        with patch.object(self.model, '_add_neighbors_to_subgraph') as mock_add:
            self.model._build_neighbor_subgraph(["node1"], 1)
            
            self.assertIsNotNone(self.model.subgraph)
            mock_add.assert_called()

    def test_extract_edges_with_attributes(self):
        """Test _extract_edges_with_attributes method."""
        self.model.subgraph = Mock()
        self.model.subgraph.get_edges.return_value = [
            ("u1", "v1", {"relation": "knows"}),
            ("u2", "v2", {"relation": "likes"})
        ]
        
        result = self.model._extract_edges_with_attributes()
        
        self.assertEqual(result, [("u1", "knows", "v1"), ("u2", "likes", "v2")])

    def test_extract_entities_from_question(self):
        """Test _extract_entities_from_question method."""
        self.mock_llm.chat.return_value = "entity1, entity2, entity3"
        
        result = self.model._extract_entities_from_question("What is entity1?")
        
        self.assertEqual(result, ["entity1", "entity2", "entity3"])
        self.mock_llm.chat.assert_called_once()

    def test_extract_entities_from_question_empty_response(self):
        """Test _extract_entities_from_question with empty entities."""
        self.mock_llm.chat.return_value = "entity1, , entity2,  "
        
        result = self.model._extract_entities_from_question("What is entity1?")
        
        self.assertEqual(result, ["entity1", "entity2"])


if __name__ == "__main__":
    unittest.main()
