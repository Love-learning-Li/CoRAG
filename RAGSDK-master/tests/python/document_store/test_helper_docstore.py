#!/usr/bin/env python3
# encoding: utf-8
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
from unittest.mock import patch, MagicMock

from sqlalchemy import URL, create_engine
from sqlalchemy.exc import SQLAlchemyError

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.storage.document_store.helper_storage import _DocStoreHelper
from mx_rag.storage.document_store.models import Base, ChunkModel
from mx_rag.utils.common import MAX_CHUNKS_NUM, STR_MAX_LEN

SQLITE = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))


class TestHelperDocStore(unittest.TestCase):
    def setUp(self):
        if os.path.exists(SQLITE):
            os.remove(SQLITE)
        # Use an in-memory SQLite database for testing
        self.url = URL.create("sqlite", database=SQLITE)
        self.engine = create_engine(self.url)
        self.docstore = _DocStoreHelper(self.engine)
        self.test_documents = [
            MxDocument(page_content="content1", metadata={"key": "value1"}, document_name="doc1"),
            MxDocument(page_content="content2", metadata={"key": "value2"}, document_name="doc2"),
        ]

    def tearDown(self):
        if os.path.exists(SQLITE):
            os.remove(SQLITE)

    def test_add_documents(self):
        doc_id = 1
        inserted_ids = self.docstore.add(self.test_documents, doc_id)
        self.assertEqual(len(inserted_ids), len(self.test_documents))
        for i, chunk_id in enumerate(inserted_ids):
            with self.docstore._transaction() as session:
                chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
                self.assertIsNotNone(chunk)
                self.assertEqual(chunk.document_id, doc_id)
                self.assertEqual(chunk.chunk_content, self.test_documents[i].page_content)
                self.assertEqual(chunk.chunk_metadata, self.test_documents[i].metadata)
                self.assertEqual(chunk.document_name, self.test_documents[i].document_name)

    def test_add_documents_with_encryption(self):
        encrypt_fn = lambda x: x + "_encrypted"
        decrypt_fn = lambda x: x[:-10] if x.endswith("_encrypted") else x
        docstore = _DocStoreHelper(self.engine, encrypt_fn=encrypt_fn, decrypt_fn=decrypt_fn)
        doc_id = 1
        docstore.add(self.test_documents, doc_id)
        with docstore._transaction() as session:
            chunks = session.query(ChunkModel).filter_by(document_id=doc_id).all()
            for chunk in chunks:
                self.assertTrue(chunk.chunk_content.endswith("_encrypted"))

    def test_add_documents_invalid_count(self):
        with self.assertRaises(ValueError):
            self.docstore.add([MxDocument(page_content="test")] * (MAX_CHUNKS_NUM + 1), 1)

    def test_delete_documents(self):
        doc_id = 1
        inserted_ids = self.docstore.add(self.test_documents, doc_id)
        deleted_ids = self.docstore.delete(doc_id)
        self.assertEqual(inserted_ids, deleted_ids)
        with self.docstore._transaction() as session:
            chunks = session.query(ChunkModel).filter_by(document_id=doc_id).all()
            self.assertEqual(len(chunks), 0)

    def test_search_document(self):
        doc_id = 1
        inserted_ids = self.docstore.add(self.test_documents, doc_id)
        retrieved_doc = self.docstore.search(inserted_ids[0])
        self.assertEqual(retrieved_doc.page_content, self.test_documents[0].page_content)
        self.assertEqual(retrieved_doc.metadata, self.test_documents[0].metadata)
        self.assertEqual(retrieved_doc.document_name, self.test_documents[0].document_name)

    def test_search_document_not_found(self):
        retrieved_doc = self.docstore.search(999)  # Non-existent ID
        self.assertIsNone(retrieved_doc)

    def test_get_all_chunk_id(self):
        doc_id = 1
        self.docstore.add(self.test_documents, doc_id)
        ids = self.docstore.get_all_chunk_id()
        self.assertEqual(len(ids), len(self.test_documents))

    @patch("mx_rag.storage.document_store.helper_storage.logger")
    def test__batch_operation_failure(self, mock_logger):
        with patch.object(self.docstore, "_transaction") as mock_transaction:
            mock_session = MagicMock()
            mock_transaction.return_value.__enter__.return_value = mock_session
            mock_session.commit.side_effect = SQLAlchemyError("Test Error")
            with self.assertRaises(StorageError):
                self.docstore._batch_operation([1, 2, 3], lambda x, s: None, "test")
            mock_logger.error.assert_called_once()

    @patch("mx_rag.storage.document_store.helper_storage.logger")
    def test__init_db_failure(self, mock_logger):
        with patch.object(Base.metadata, "create_all") as mock_create_all:
            mock_create_all.side_effect = SQLAlchemyError("Test Error")
            with self.assertRaises(StorageError):
                _DocStoreHelper(self.engine)
            mock_logger.critical.assert_called_once()

    def test_update(self):
        with self.assertRaises(StorageError):
            self.docstore.update([1, 2], ["text1", "text2"])

    def test_mx_document(self):
        MxDocument(page_content="hello", metadata={}, document_name="name")
        with self.assertRaises(ValueError):
            MxDocument(page_content="hello", metadata={"key": "a" * (STR_MAX_LEN+1)}, document_name="name")


if __name__ == '__main__':
    unittest.main()
