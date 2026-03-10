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
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool

from mx_rag.storage.vectorstore.opengauss import (
    OpenGaussDB,
    SearchMode,
    OpenGaussError,
    StorageError
)


class TestOpenGaussDB(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock pool
        self.mock_pool = Mock(spec=Pool)
        self.mock_pool.size = Mock(return_value=8)

        # Create mock engine with proper attributes
        self.mock_engine = Mock(spec=Engine)
        self.mock_engine.pool = self.mock_pool

        # Create a mock dialect with has_table method
        self.mock_dialect = Mock()
        self.mock_dialect.has_table.return_value = False
        self.mock_engine.dialect = self.mock_dialect
        self.mock_engine.name = "opengauss"

        # Create a mock connection
        self.mock_connection = Mock()
        self.mock_engine.connect.return_value = self.mock_connection

        self.mock_session = MagicMock(spec=Session)

        self.db = OpenGaussDB(
            engine=self.mock_engine,
            collection_name="test_collection",
            search_mode=SearchMode.DENSE
        )

    def test_init_validates_params(self):
        """Test parameter validation during initialization."""
        with self.assertRaises(ValueError):
            OpenGaussDB(engine="not_an_engine")

        with self.assertRaises(ValueError):
            OpenGaussDB(engine=Mock(spec=Engine), collection_name="a" * 2000)

    def test_create_collection(self):
        """Test collection creation with various parameters."""
        # Test successful creation
        with patch('mx_rag.storage.vectorstore.opengauss.Base.metadata.create_all'):
            self.db.create_collection(dense_dim=128)
            self.assertIsNotNone(self.db.vector_model)

        # Test missing dense_dim for DENSE mode
        with self.assertRaises(OpenGaussError):
            self.db.create_collection()

    def test_add(self):
        """Test adding dense vectors."""
        self.db.create_collection(dense_dim=3)
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ids = [1, 2]

        with patch.object(self.db, '_internal_add') as mock_add:
            self.db.add(ids, embeddings, document_id=0)
            mock_add.assert_called_once_with(ids, embeddings, document_id=0)

    def test_add_sparse(self):
        """Test adding sparse vectors."""
        db = OpenGaussDB(self.mock_engine, search_mode=SearchMode.SPARSE)
        db.create_collection(sparse_dim=100)
        sparse_embeddings = [
            {1: 0.5, 2: 0.3},
            {2: 0.7, 3: 0.2}
        ]
        ids = [1, 2]

        with patch.object(db, '_internal_add') as mock_add:
            db.add_sparse(ids, sparse_embeddings)
            mock_add.assert_called_once_with(ids, sparse=sparse_embeddings, document_id=0)

    def test_delete(self):
        """Test deleting vectors."""
        self.db.create_collection(dense_dim=128)
        with patch.object(self.db, '_transaction') as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = self.mock_session
            self.mock_session.query().filter().delete.return_value = 2

            result = self.db.delete([1, 2])
            self.assertEqual(result, 2)

    def test_search(self):
        """Test vector search functionality."""
        self.db.create_collection(dense_dim=3)
        query_vectors = [[1.0, 2.0, 3.0]]

        with patch.object(self.db, '_parallel_search') as mock_search:
            mock_search.return_value = ([0.9], [1])
            scores, ids = self.db.search(query_vectors, k=1)

            self.assertEqual(scores, [0.9])
            self.assertEqual(ids, [1])
            mock_search.assert_called_once_with(query_vectors, 1)

    def test_get_all_ids(self):
        """Test retrieving all vector IDs."""
        self.db.create_collection(dense_dim=128)
        self.mock_session.query().all.return_value = [(1,), (2,), (3,)]

        with patch.object(self.db, '_transaction') as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = self.mock_session
            ids = self.db.get_all_ids()
            self.assertEqual(ids, [1, 2, 3])

    def test_drop_collection(self):
        """Test collection dropping functionality."""
        # Mock session and execution results
        mock_session = MagicMock()
        mock_session.execute.return_value.fetchall.return_value = [
            ('index1',), ('index2',)  # Mock indexes found
        ]

        # Create mock metadata and table
        mock_table = Mock()
        mock_metadata = Mock()
        mock_metadata.tables = {"test_collection": mock_table}

        # Mock the identifier preparer
        mock_preparer = Mock()
        mock_preparer.quote_identifier.return_value = '"test_collection"'
        self.mock_engine.dialect.identifier_preparer = mock_preparer

        with patch.object(self.db, '_transaction') as mock_transaction, \
                patch('mx_rag.storage.vectorstore.opengauss.MetaData') as mock_metadata_class:
            # Set up the mock transaction to return our mock session
            mock_transaction.return_value.__enter__.return_value = mock_session

            # Set up the mock metadata
            mock_metadata_class.return_value = mock_metadata

            # Call drop_collection
            self.db.drop_collection()

            # Verify that:
            # 1. Table name was properly quoted
            mock_preparer.quote_identifier.assert_called_once_with("test_collection")

            # 2. Session queried for indexes with safe parameter binding
            query_call = mock_session.execute.call_args_list[0]
            self.assertIn('SELECT indexname', str(query_call[0][0]))
            self.assertIn('FROM pg_indexes', str(query_call[0][0]))
            self.assertEqual(query_call[0][1], {"table_name": '"test_collection"'})

            # 3. Each index was dropped
            drop_calls = mock_session.execute.call_args_list[1:]
            self.assertIn('DROP INDEX IF EXISTS index1', str(drop_calls[0][0][0]))
            self.assertIn('DROP INDEX IF EXISTS index2', str(drop_calls[1][0][0]))

            # 4. Metadata was reflected
            mock_metadata.reflect.assert_called_once_with(bind=self.mock_engine)

            # 5. Table was dropped
            mock_table.drop.assert_called_once_with(self.mock_engine, checkfirst=True)

            # 6. Metadata was cleared
            mock_metadata.clear.assert_called_once()

    def test_drop_collection_invalid_table_name(self):
        """Test drop_collection with invalid table name."""
        # Create instance with invalid table name
        with self.assertRaises(ValueError):
            _ = OpenGaussDB(
                engine=self.mock_engine,
                collection_name="invalid;name",  # Invalid identifier
                search_mode=SearchMode.DENSE
            )

    def test_add_dense_and_sparse(self):
        """Test adding both dense and sparse vectors in hybrid mode."""
        db = OpenGaussDB(self.mock_engine, search_mode=SearchMode.HYBRID)
        db.create_collection(dense_dim=3, sparse_dim=100)

        dense_embeddings = np.array([[1.0, 2.0, 3.0]])
        sparse_embeddings = [{1: 0.5, 2: 0.3}]
        ids = [1]

        with patch.object(db, '_internal_add') as mock_add:
            db.add_dense_and_sparse(ids, dense_embeddings, sparse_embeddings, document_id=0)
            mock_add.assert_called_once_with(ids, dense_embeddings, sparse_embeddings, 0)

    def test_invalid_search_modes(self):
        """Test invalid search mode combinations."""
        # Test adding dense vectors in SPARSE mode
        db = OpenGaussDB(self.mock_engine, search_mode=SearchMode.SPARSE)
        with self.assertRaises(ValueError):
            db.add([1], np.array([[1.0, 2.0]]))

        # Test adding sparse vectors in DENSE mode
        db = OpenGaussDB(self.mock_engine, search_mode=SearchMode.DENSE)
        with self.assertRaises(ValueError):
            db.add_sparse([1], [{1: 0.5}])

    def test_create_class_method(self):
        """Test the create class method."""
        with patch.object(OpenGaussDB, 'create_collection') as mock_create:
            instance = OpenGaussDB.create(
                engine=self.mock_engine,
                collection_name="test",
                dense_dim=128,
            )
            self.assertIsInstance(instance, OpenGaussDB)
            mock_create.assert_called_once()

    def test_create_class_method_error(self):
        """Test the create class method with missing required parameters."""
        instance = OpenGaussDB.create()
        self.assertIsNone(instance)

    def test_parallel_search(self):
        """Test the parallel search functionality."""
        self.db.create_collection(dense_dim=3)
        query_vectors = [[1.0, 2.0, 3.0]]

        # Mock the _do_search method to return some results
        with patch.object(self.db, '_do_search') as mock_do_search:
            mock_do_search.return_value = ([Mock(id=1)], [0.9])

            scores, ids = self.db._parallel_search(query_vectors, k=1)
            self.assertEqual(len(scores), 1)
            self.assertEqual(len(ids), 1)

    def test_calculate_pool_size(self):
        """Test the pool size calculation."""
        with patch('multiprocessing.cpu_count', return_value=8):
            pool_size = self.db._calculate_pool_size()
            self.assertIsInstance(pool_size, int)
            self.assertGreater(pool_size, 0)
            # Verify that pool.size() was called
            self.mock_pool.size.assert_called_once()

    def test_parallel_search_with_pool_size(self):
        """Test parallel search with specific pool size configuration."""
        self.db.create_collection(dense_dim=3)
        query_vectors = [[1.0, 2.0, 3.0]]

        # Create a mock result that matches the expected structure
        mock_result = Mock()
        mock_result.id = 1
        mock_results = ([mock_result], [0.9])  # Tuple of (results, scores)

        # Mock ThreadPool
        mock_pool = MagicMock()
        mock_pool.__enter__.return_value = mock_pool
        mock_pool.starmap.return_value = [mock_results]  # List of results for each query

        with patch('mx_rag.storage.vectorstore.opengauss.ThreadPool', return_value=mock_pool) as mock_thread_pool:
            with patch.object(self.db, '_do_search', return_value=mock_results):
                scores, ids = self.db._parallel_search(query_vectors, k=1)

                # Verify results
                self.assertEqual(scores, [[-0.9]])  # Note: scores are scaled by similarity strategy
                self.assertEqual(ids, [[1]])

                # Verify pool was created with correct size
                mock_thread_pool.assert_called_once()

                # Verify starmap was called
                mock_pool.starmap.assert_called_once()

    def test_do_search(self):
        """Test the individual search operation."""
        self.db.create_collection(dense_dim=3)
        query_vector = np.array([1.0, 2.0, 3.0])

        # Mock the session query results
        mock_result = Mock()
        mock_result.id = 1

        with patch.object(self.db, '_transaction') as mock_transaction:
            mock_session = MagicMock()
            mock_session.query.return_value.order_by.return_value.params.return_value.limit. \
                return_value.all.return_value = [(mock_result, 0.9)]
            mock_transaction.return_value.__enter__.return_value = mock_session

            results, scores = self.db._do_search(query_vector, k=1, metric_func_op="<->")

            self.assertEqual(len(results), 1)
            self.assertEqual(len(scores), 1)
            self.assertEqual(results[0].id, 1)
            self.assertEqual(scores[0], 0.9)

    def test_search_with_empty_vectors(self):
        """Test search with empty input vectors."""
        self.db.create_collection(dense_dim=3)

        # Test with empty list
        with self.assertRaises(ValueError):
            _ = self.db.search([], k=1)

    def test_search_with_multiple_vectors(self):
        """Test search with multiple input vectors."""
        self.db.create_collection(dense_dim=3)
        query_vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        # Mock the parallel search results
        mock_results = [
            ([Mock(id=1)], [0.9]),
            ([Mock(id=2)], [0.8])
        ]

        with patch.object(self.db, '_do_search') as mock_do_search:
            mock_do_search.side_effect = mock_results

            with patch('multiprocessing.pool.ThreadPool') as mock_pool:
                mock_pool.return_value.__enter__.return_value.starmap.return_value = mock_results

                scores, ids = self.db.search(query_vectors, k=1)

                self.assertEqual(len(scores), 2)
                self.assertEqual(len(ids), 2)
                # Verify scores are scaled according to similarity strategy
                self.assertEqual(scores[0], [-0.9])
                self.assertEqual(scores[1], [-0.8])

    def test_search_with_invalid_k(self):
        """Test search with invalid k parameter."""
        self.db.create_collection(dense_dim=3)
        query_vectors = [[1.0, 2.0, 3.0]]

        with self.assertRaises(ValueError):
            self.db.search(query_vectors, k=0)

        with self.assertRaises(ValueError):
            self.db.search(query_vectors, k=11000)

    def test_transaction_context_manager(self):
        """Test the transaction context manager."""
        self.db.create_collection(dense_dim=3)

        # Test successful transaction
        mock_session = MagicMock()

        with patch.object(self.db, 'session_factory') as mock_session_factory:
            # Set up the session factory to return our mock session
            mock_session_factory.return_value = mock_session

            # Use the transaction context manager
            with self.db._transaction() as session:
                # Just verify we got the same session back
                self.assertIs(session, mock_session)

            # Verify the session operations
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

        # Test failed transaction
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("Database error")

        with patch.object(self.db, 'session_factory') as mock_session_factory:
            mock_session_factory.return_value = mock_session

            with self.assertRaises(StorageError):
                with self.db._transaction():
                    pass  # The commit will fail

            # Verify error handling
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()

    def test_prepare_insert_data(self):
        """Test the preparation of insert data."""
        self.db.create_collection(dense_dim=3, sparse_dim=100)

        # Test dense data preparation
        dense_data = np.array([[1.0, 2.0, 3.0]])
        ids = [1]
        result = self.db._prepare_insert_data(ids, dense=dense_data)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["vector"], [1.0, 2.0, 3.0])

        # Test sparse data preparation
        sparse_data = [{1: 0.5, 2: 0.3}]
        result = self.db._prepare_insert_data(ids, sparse=sparse_data)
        self.assertEqual(result[0]["id"], 1)
        self.assertIn("sparse_vector", result[0])

        # Test mismatched lengths
        with self.assertRaises(ValueError):
            self.db._prepare_insert_data([1, 2], dense=dense_data)

    def test_update(self):
        dense_data = np.array([[1.0, 2.0, 3.0]])
        sparse_data = [{1: 0.5, 2: 0.3}]
        with self.assertRaises(ValueError):
            self.db.update([1, 2, 3], dense_data, sparse_data)
        self.db.sparse_dim = 1
        self.db._get_vec_by_id = lambda x: [{"id": 1, "vector": [0.1], "sparse_vector": [{}]},
                                            {"id": 2, "vector": [0.1], "sparse_vector": [{}]},
                                            {"id": 3, "vector": [0.1], "sparse_vector": [{}]}]
        with self.assertRaises(StorageError):
            self.db.update([1, 2, 3], np.array([[1.0], [2.0], [3.0]]), [{1: 0.5}, {2: 0.3}, {3: 0.4}])

    def test_fake_engine(self):
        mock_engine = MagicMock(spec=Engine)
        mock_engine.name = "mysql"
        with self.assertRaises(StorageError):
            OpenGaussDB(engine=mock_engine)


if __name__ == '__main__':
    unittest.main()
