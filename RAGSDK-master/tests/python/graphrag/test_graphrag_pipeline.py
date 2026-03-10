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
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from mx_rag.graphrag.graphrag_pipeline import GraphRAGPipeline, GraphRetriever
from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.reranker.reranker import Reranker
from mx_rag.document.doc_loader_mng import LoaderMng
from mx_rag.utils.common import Lang
from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.graphrag.graph_rag_model import GraphRAGModel


class TestGraphRetriever(unittest.TestCase):
    """Test the GraphRetriever class."""
    
    def setUp(self):
        self.mock_graph_rag_model = Mock(spec=GraphRAGModel)
        self.mock_graph_rag_model.generate.return_value = [["result1", "result2"]]
        self.retriever = GraphRetriever(graph_rag_model=self.mock_graph_rag_model)
    
    def test_get_relevant_documents(self):
        """Test the _get_relevant_documents method."""
        query = "test query"
        result = self.retriever._get_relevant_documents(query)
        
        self.assertEqual(result, ["result1", "result2"])
        self.mock_graph_rag_model.generate.assert_called_once_with([query])
    
    def test_get_relevant_documents_validation_error(self):
        """Test validation error for invalid query."""
        with self.assertRaises(Exception):  # Should raise validation error
            self.retriever._get_relevant_documents("")  # Empty string
        
        with self.assertRaises(Exception):  # Should raise validation error
            self.retriever._get_relevant_documents("x" * 1000001)  # Too long


class TestGraphRAGPipeline(unittest.TestCase):
    """Test the GraphRAGPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/tmp"))
        os.makedirs(self.temp_dir)
        # Mock dependencies
        self.mock_llm = Mock(spec=Text2TextLLM)
        self.mock_llm.llm_config = Mock()
        self.mock_embedding_model = Mock(spec=Embeddings)
        self.mock_rerank_model = Mock(spec=Reranker)
        
        # Mock embedding model methods
        self.mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        self.test_dim = 768
        self.test_graph_name = "test_graph"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    def test_init_success_networkx(self, mock_check_space):
        """Test successful initialization with networkx graph."""
        mock_check_space.return_value = False  # Sufficient space
        
        pipeline = GraphRAGPipeline(
            work_dir=self.temp_dir,
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            rerank_model=self.mock_rerank_model,
            dim=self.test_dim,
            graph_type="networkx",
            graph_name=self.test_graph_name
        )
        
        self.assertEqual(pipeline.work_dir, self.temp_dir)
        self.assertEqual(pipeline.graph_name, self.test_graph_name)
        self.assertEqual(pipeline.llm, self.mock_llm)
        self.assertEqual(pipeline.embedding_model, self.mock_embedding_model)
        self.assertEqual(pipeline.rerank_model, self.mock_rerank_model)
        self.assertEqual(pipeline.dim, self.test_dim)
        self.assertEqual(pipeline.docs, [])
        self.assertIsNone(pipeline.concept_embedding)
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    def test_init_insufficient_disk_space(self, mock_check_space):
        """Test initialization failure due to insufficient disk space."""
        mock_check_space.return_value = True  # Insufficient space
        
        with self.assertRaises(StorageError):
            GraphRAGPipeline(
                work_dir=self.temp_dir,
                llm=self.mock_llm,
                embedding_model=self.mock_embedding_model,
                rerank_model=self.mock_rerank_model,
                dim=self.test_dim
            )
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    def test_setup_save_path(self, mock_check_space):
        """Test _setup_save_path method."""
        mock_check_space.return_value = False
        
        pipeline = GraphRAGPipeline(
            work_dir=self.temp_dir,
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            rerank_model=self.mock_rerank_model,
            dim=self.test_dim,
            graph_name=self.test_graph_name
        )
        
        expected_graph_path = os.path.join(self.temp_dir, f"{self.test_graph_name}.json")
        expected_relations_path = os.path.join(self.temp_dir, f"{self.test_graph_name}_relations.json")
        expected_concepts_path = os.path.join(self.temp_dir, f"{self.test_graph_name}_concepts.json")
        expected_synset_path = os.path.join(self.temp_dir, f"{self.test_graph_name}_synset.json")
        expected_node_vectors_path = os.path.join(self.temp_dir, f"{self.test_graph_name}_node_vectors.index")
        expected_concept_vectors_path = os.path.join(self.temp_dir, f"{self.test_graph_name}_concept_vectors.index")
        
        self.assertEqual(pipeline.graph_save_path, expected_graph_path)
        self.assertEqual(pipeline.relations_save_path, expected_relations_path)
        self.assertEqual(pipeline.concepts_save_path, expected_concepts_path)
        self.assertEqual(pipeline.synset_save_path, expected_synset_path)
        self.assertEqual(pipeline.node_vectors_path, expected_node_vectors_path)
        self.assertEqual(pipeline.concept_vectors_path, expected_concept_vectors_path)
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    def test_upload_files_success(self, mock_check_space):
        """Test successful file upload."""
        mock_check_space.return_value = False
        
        pipeline = GraphRAGPipeline(
            work_dir=self.temp_dir,
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            rerank_model=self.mock_rerank_model,
            dim=self.test_dim
        )
        
        # Create temporary test files
        test_file1 = os.path.join(self.temp_dir, "test1.txt")
        test_file2 = os.path.join(self.temp_dir, "test2.txt")
        with open(test_file1, 'w') as f:
            f.write("test content 1")
        with open(test_file2, 'w') as f:
            f.write("test content 2")
        
        # Mock loader manager
        mock_loader_mng = Mock(spec=LoaderMng)
        mock_loader_info = Mock()
        mock_loader_class = Mock()
        mock_loader_instance = Mock()
        mock_splitter_info = Mock()
        mock_splitter_class = Mock()
        mock_splitter_instance = Mock()
        
        mock_loader_info.loader_class = mock_loader_class
        mock_loader_info.loader_params = {}
        mock_splitter_info.splitter_class = mock_splitter_class
        mock_splitter_info.splitter_params = {}
        
        mock_loader_class.return_value = mock_loader_instance
        mock_splitter_class.return_value = mock_splitter_instance
        mock_loader_instance.load_and_split.return_value = [
            Document(page_content="content1"),
            Document(page_content="content2")
        ]
        
        mock_loader_mng.get_loader.return_value = mock_loader_info
        mock_loader_mng.get_splitter.return_value = mock_splitter_info
        
        with patch('mx_rag.graphrag.graphrag_pipeline.FileCheck.check_path_is_exist_and_valid'):
            pipeline.upload_files([test_file1, test_file2], mock_loader_mng)
        
        # Verify documents were added
        self.assertEqual(len(pipeline.docs), 4)  # 2 files × 2 documents each
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    @patch('mx_rag.graphrag.graphrag_pipeline.write_to_json')
    @patch('mx_rag.graphrag.graphrag_pipeline.LLMRelationExtractor')
    @patch('mx_rag.graphrag.graphrag_pipeline.GraphMerger')
    @patch('mx_rag.graphrag.graphrag_pipeline.logger')
    def test_build_graph_success(self, mock_logger, mock_graph_merger, mock_extractor_class,
                                 write_to_json, mock_check_space):
        """Test successful graph building."""
        mock_check_space.return_value = False
        
        pipeline = GraphRAGPipeline(
            work_dir=self.temp_dir,
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            rerank_model=self.mock_rerank_model,
            dim=self.test_dim
        )
        
        # Add some test documents
        docs = [Document(page_content="test content")]
        pipeline.docs = docs
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.query.return_value = [{"entity1": "relation", "entity2": "data"}]
        mock_extractor_class.return_value = mock_extractor
        
        # Mock merger
        mock_merger = Mock()
        mock_graph_merger.return_value = mock_merger
        
        pipeline.build_graph(lang=Lang.EN, pad_token="<pad>", conceptualize=False)
        
        # Verify extractor was called
        mock_extractor_class.assert_called_once()
        mock_extractor.query.assert_called_once_with(docs)
        
        # Verify merger was called
        mock_graph_merger.assert_called_once()
        mock_merger.merge.assert_called_once()
        mock_merger.save_graph.assert_called_once()
        
        # Verify JSON was saved
        write_to_json.assert_called_once()
        
        # Verify docs were cleared
        self.assertEqual(len(pipeline.docs), 0)
        
        # Verify success message
        mock_logger.info.assert_called_with("Graph built successfully")
    
    @patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space')
    @patch('mx_rag.graphrag.graphrag_pipeline.VectorStorageFactory')
    @patch('mx_rag.graphrag.graphrag_pipeline.VectorStoreWrapper')
    @patch('mx_rag.graphrag.graphrag_pipeline.GraphRAGModel')
    def test_as_retriever_success(self, mock_graph_rag_model, mock_wrapper, mock_storage_factory, mock_check_space):
        """Test successful as_retriever method."""
        mock_check_space.return_value = False
        
        pipeline = GraphRAGPipeline(
            work_dir=self.temp_dir,
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            rerank_model=self.mock_rerank_model,
            dim=self.test_dim
        )
        
        # Mock storage factory
        mock_node_store = Mock()
        mock_concept_store = Mock()
        mock_storage_factory.create_storage.side_effect = [mock_node_store, mock_concept_store]
        
        # Mock wrapper
        mock_node_wrapper = Mock()
        mock_concept_wrapper = Mock()
        mock_wrapper.side_effect = [mock_node_wrapper, mock_concept_wrapper]
        
        # Mock GraphRAGModel
        mock_rag_model = Mock(spec=GraphRAGModel)
        mock_graph_rag_model.return_value = mock_rag_model
        
        retriever = pipeline.as_retriever()
        
        # Verify retriever was created
        self.assertIsInstance(retriever, GraphRetriever)
        self.assertEqual(retriever.graph_rag_model, mock_rag_model)
        
        # Verify GraphRAGModel was initialized
        mock_graph_rag_model.assert_called_once()
    
    def test_as_retriever_invalid_parameters(self):
        """Test as_retriever with invalid parameters."""
        with patch('mx_rag.graphrag.graphrag_pipeline.check_disk_free_space', return_value=False), \
             patch('mx_rag.graphrag.graphrag_pipeline.FileCheck.check_input_path_valid'), \
             patch('mx_rag.graphrag.graphrag_pipeline.FileCheck.check_filename_valid'):
            
            pipeline = GraphRAGPipeline(
                work_dir=self.temp_dir,
                llm=self.mock_llm,
                embedding_model=self.mock_embedding_model,
                rerank_model=self.mock_rerank_model,
                dim=self.test_dim
            )


if __name__ == '__main__':
    unittest.main()