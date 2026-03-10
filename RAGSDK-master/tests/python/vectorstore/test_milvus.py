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
from unittest.mock import patch, MagicMock
import numpy as np
from pymilvus import MilvusClient
from pymilvus.client.types import ExtraList

from mx_rag.storage.vectorstore.milvus import MilvusError
from mx_rag.storage.vectorstore.vectorstore import VectorStore, SearchMode
from mx_rag.storage.vectorstore import MilvusDB


class TestMilvusDB(unittest.TestCase):
    def test_faiss(self):
        with patch("pymilvus.MilvusClient") as MilvusClient:
            embeddings = np.random.random((3, 1024))
            query = embeddings[0]
            my_milvus = MagicMock()
            my_milvus.set_collection_name("test_ccc")
            my_milvus.client.has_collection = MagicMock(return_value=False)
            my_milvus.create_collection(768, "FLAT", "IP")
            my_milvus.client.has_collection = MagicMock(return_value=True)
            my_milvus.add([0, 1, 2], embeddings)
            res = my_milvus.search(query)
            my_milvus.delete([0, 1, 2])
            my_milvus.drop_collection()

    def setUp(self):
        self.client = MagicMock(spec=MilvusClient)
        self.dense_kwargs = dict(
            client=self.client,
            x_dim=1024,
            collection_name="dense_collection"
        )
        self.sparse_kwargs = dict(
            client=self.client,
            collection_name="sparse_collection",
            search_mode=SearchMode.SPARSE,
            index_type="HNSW",
            metric_type="IP"
        )
        self.hybrid_kwargs = dict(
            client=self.client,
            collection_name="hybrid_collection",
            search_mode=SearchMode.HYBRID,
            x_dim=1024
        )
        self.not_exist_collection = "not_exist_collection"
        self.sparse_vecs = [{1: 0.1, 2: 0.3}, {1: 0.2, 2: 0.1, 3: 0.2}]
        self.dense_vecs = np.random.randn(2, 1024)
        self.ids = [1, 2]
        self.docs = ["rag", "sdk"]

    def create_milvus_db_dense(self):
        return MilvusDB.create(**self.dense_kwargs)

    def create_milvus_db_sparse(self):
        return MilvusDB.create(**self.sparse_kwargs)

    def create_milvus_db_hybrid(self):
        return MilvusDB.create(**self.hybrid_kwargs)

    def test_create_no_client(self):
        del self.dense_kwargs['client']
        milvus_db = MilvusDB.create(**self.dense_kwargs)
        self.assertIsNone(milvus_db)

    def test_create_no_x_dim(self):
        del self.dense_kwargs['x_dim']
        milvus_db = MilvusDB.create(**self.dense_kwargs)
        self.assertIsNotNone(milvus_db)

    def test_create_no_similarity_strategy(self):
        milvus_db = MilvusDB.create(**self.dense_kwargs)
        self.assertIsNotNone(milvus_db)

    def test_create_wrong_params(self):
        milvus_db = MilvusDB.create(**self.dense_kwargs, params="params")
        self.assertIsNone(milvus_db)

    def test_create_success(self):
        milvus_db = MilvusDB.create(**self.dense_kwargs)
        self.assertIsInstance(milvus_db, MilvusDB)
        self.assertEqual(milvus_db._collection_name, self.dense_kwargs['collection_name'])

    def test_add_data(self):
        vecs = np.random.randn(3, 1024)
        self.create_milvus_db_dense().add([0, 1, 2], vecs)
        self.create_milvus_db_dense().client.insert.assert_called_once()
        self.create_milvus_db_dense().client.refresh_load.assert_called_once()

        with self.assertRaises(ValueError):
            vecs = np.random.randn(3, 2, 1024)
            self.create_milvus_db_dense().add([0, 1, 2], vecs)

        with self.assertRaises(MilvusError):
            vecs = np.random.randn(2, 1024)
            self.create_milvus_db_dense().add([0, 1, 2], vecs)

        vecs = np.random.randn(3, 1024)
        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db_dense().add([0, 1, 2], vecs)

        with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
            with self.assertRaises(MilvusError):
                self.create_milvus_db_dense().add([0, 1, 2], vecs)

    def test_add_sparse_no_docs(self):
        db = self.create_milvus_db_sparse()
        db.add_sparse(self.ids, self.sparse_vecs)
        self.create_milvus_db_dense().client.insert.assert_called_once()
        self.create_milvus_db_dense().client.refresh_load.assert_called_once()

    def test_add_sparse_with_docs(self):
        db = self.create_milvus_db_sparse()
        db.add_sparse(self.ids, self.sparse_vecs, docs = self.docs)
        self.create_milvus_db_dense().client.insert.assert_called_once()
        self.create_milvus_db_dense().client.refresh_load.assert_called_once()

    def test_add_sparse_collection_not_exist(self):
        db = self.create_milvus_db_sparse()
        db._validate_collection_existence = MagicMock(side_effect=MilvusError("Collection does not exist"))
        with self.assertRaises(MilvusError):
            db.add_sparse(self.ids, self.sparse_vecs)

    def test_add_sparse_invalid_search_mode(self):
        db = self.create_milvus_db_sparse()
        db._search_mode = SearchMode.DENSE
        with self.assertRaises(MilvusError):
            db.add_sparse(self.ids, self.sparse_vecs)

    def test_add_dense_and_sparse_no_docs(self):
        db = self.create_milvus_db_hybrid()
        db.add_dense_and_sparse(self.ids, self.dense_vecs, self.sparse_vecs)
        self.create_milvus_db_dense().client.insert.assert_called_once()
        self.create_milvus_db_dense().client.refresh_load.assert_called_once()

    def test_add_dense_and_sparse_with_docs(self):
        db = self.create_milvus_db_hybrid()
        db.add_dense_and_sparse(self.ids, self.dense_vecs, self.sparse_vecs, self.docs)
        self.create_milvus_db_dense().client.insert.assert_called_once()
        self.create_milvus_db_dense().client.refresh_load.assert_called_once()

    def test_add_dense_and_sparse_collection_not_exist(self):
        db = self.create_milvus_db_hybrid()
        db._validate_collection_existence = MagicMock(side_effect=MilvusError("Collection does not exist"))
        with self.assertRaises(MilvusError):
            db.add_dense_and_sparse(self.ids, self.dense_vecs, self.sparse_vecs)

    def test_add_dense_and_sparse_invalid_search_mode(self):
        db = self.create_milvus_db_hybrid()
        db._search_mode = SearchMode.DENSE
        with self.assertRaises(MilvusError):
            db.add_dense_and_sparse(self.ids, self.dense_vecs, self.sparse_vecs)

    def test_delete_data(self):
        vecs = np.random.randn(3, 1024)
        self.create_milvus_db_dense().add([0, 1, 2], vecs)
        with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
            with self.assertRaises(MilvusError):
                self.create_milvus_db_dense().delete([1, 2])

        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db_dense().delete([0])

        with self.assertRaises(ValueError):
            self.create_milvus_db_dense().delete(['0'])

        self.create_milvus_db_dense().client.delete.return_value = {'delete_count': len(vecs)}
        self.create_milvus_db_dense().client.has_collection.return_value = True
        result = self.create_milvus_db_dense().delete([0])
        self.assertEqual(result, len(vecs))

    def test_get_all_ids(self):
        ids = self.create_milvus_db_dense().get_all_ids()
        self.assertEqual(ids, [])

    def test_drop_collection(self):
        self.create_milvus_db_dense().drop_collection()
        self.create_milvus_db_dense().client.drop_collection.assert_called_once_with(
            self.create_milvus_db_dense()._collection_name)

    def test_search(self):
        with patch('mx_rag.storage.vectorstore.vectorstore.VectorStore._score_scale') as mock_score_scale:
            mock_score_scale.return_value = [1, 2, 3]
            self.client.search.return_value = [
                [{'distance': 0.1, 'id': 1}, {'distance': 0.2, 'id': 2}, {'distance': 0.3, 'id': 3}],
                [{'distance': 0.4, 'id': 4}, {'distance': 0.5, 'id': 5}, {'distance': 0.6, 'id': 6}]
            ]
            embedding = np.array([[1, 2, 3], [4, 5, 6]])
            scores, ids = self.create_milvus_db_dense().search(embedding.tolist(), 3)[:2]
            self.assertEqual(scores, [1, 2, 3])
            self.assertEqual(ids, [[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            embedding = np.random.randn(3, 2, 1024)
            self.create_milvus_db_dense().search(embedding.tolist(), 3)

        with patch.object(VectorStore, 'MAX_SEARCH_BATCH', 1):
            with self.assertRaises(ValueError):
                self.create_milvus_db_dense().search(np.array([[1, 2], [4, 5]]).tolist())

        with self.assertRaises(MilvusError):
            self.client.has_collection.return_value = False
            self.create_milvus_db_dense().search(np.array([[1, 2]]).tolist())

    @patch("mx_rag.storage.vectorstore.milvus.MilvusDB.client")
    def test_update(self, client_mocker):
        client_mocker.get.return_value = [{"id": 1, "vector": self.dense_vecs[0], "sparse_vector": self.sparse_vecs[0]},
                                          {"id": 2, "vector": self.dense_vecs[0], "sparse_vector": self.sparse_vecs[0]}]
        self.create_milvus_db_dense().update(self.ids, self.dense_vecs, self.sparse_vecs)

    @patch("pymilvus.orm.connections.Connections")
    def test_set_collection_name(self, connection_mocker):
        db = self.create_milvus_db_dense()
        db.set_collection_name("hello")
        self.assertEqual("hello", db.collection_name)
        db._search_mode = SearchMode.DENSE
        connection_mocker.has_collection.return_value = "hello"
        self.assertTrue(db.has_collection("hello"))

    def test_validate_docs(self):
        db = self.create_milvus_db_dense()
        with self.assertRaises(MilvusError):
            db._validate_metadatas([[1]])
        with self.assertRaises(MilvusError):
            db._validate_metadatas([1] * 1024 * 1025)
        with self.assertRaises(ValueError):
            db._validate_docs(["hello", 1])

    def test_perform_search(self):
        db = self.create_milvus_db_dense()
        with self.assertRaises(ValueError):
            db._search_mode = SearchMode.SPARSE
            db._perform_dense_search(np.array([1]), 1, [])
        with self.assertRaises(ValueError):
            db._search_mode = SearchMode.DENSE
            db._perform_sparse_search([{1: 0.1}], 1, [])
        data = ExtraList([[{"id": 1, "distance": 0.1}]], extra={"total": 3})
        scores, ids, extras = db._process_search_results(data)
        self.assertEqual(scores, [[0.95]])
        self.assertEqual(ids, [[1]])
        self.assertEqual(extras, [[]])

    def test_init_insert_data(self):
        db = self.create_milvus_db_dense()
        data = db._init_insert_data([1], ["text"], [{1: 0.1}], 1)
        self.assertEqual(data, [{'id': 1, 'document_id': 1, 'document': 'text', 'metadata': {1: 0.1}}])


if __name__ == "__main__":
    unittest.main()
