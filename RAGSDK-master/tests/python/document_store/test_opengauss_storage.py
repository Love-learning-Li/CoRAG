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

from sqlalchemy import Engine

from mx_rag.storage.document_store.base_storage import MxDocument, StorageError
from mx_rag.storage.document_store.helper_storage import _DocStoreHelper
from mx_rag.storage.document_store.opengauss_storage import OpenGaussDocstore
from mx_rag.utils.common import MAX_CHUNKS_NUM


class TestOpenGaussDocstore(unittest.TestCase):
    @patch("mx_rag.storage.document_store.opengauss_storage._DocStoreHelper")
    @patch('sqlalchemy.create_engine')
    def setUp(self, mock_create_engine, MockDocStoreHelper):
        # Mock the engine and connection
        self.mock_engine = MagicMock(spec=Engine)
        self.mock_engine.name = "opengauss"
        mock_create_engine.return_value = self.mock_engine  # Make create_engine return the mock_engine
        self.mock_helper = MagicMock(spec=_DocStoreHelper)  # mock HelperDocStore
        MockDocStoreHelper.return_value = self.mock_helper
        self.docstore = OpenGaussDocstore(engine=self.mock_engine, enable_bm25=False)
        self.test_documents = [
            MxDocument(page_content="content1", document_name="test1.docx", metadata={"key": "value1"}),
            MxDocument(page_content="content2", document_name="test1.docx", metadata={"key": "value2"}),
        ]

    def test_add_documents(self):
        doc_id = 1
        expected_ids = [1, 2]
        self.mock_helper.add.return_value = expected_ids

        returned_ids = self.docstore.add(self.test_documents, doc_id)

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.add.assert_called_once_with(self.test_documents, doc_id)

    def test_add_documents_invalid_input(self):
        with self.assertRaises(ValueError):
            self.docstore.add([1, 2, 3], 1)  # Invalid document type

        with self.assertRaises(ValueError):
            self.docstore.add([], 1)  # Empty list

        with self.assertRaises(ValueError):
            self.docstore.add([MxDocument(page_content="test")] * (MAX_CHUNKS_NUM + 1), 1)  # Too many documents

    def test_delete_documents(self):
        doc_id = 1
        expected_ids = [1, 2]
        self.mock_helper.delete.return_value = expected_ids

        returned_ids = self.docstore.delete(doc_id)

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.delete.assert_called_once_with(doc_id)

    def test_search_document(self):
        chunk_id = 1
        expected_doc = MxDocument(page_content="test", document_name="test1.docx", metadata={})
        self.mock_helper.search.return_value = expected_doc

        returned_doc = self.docstore.search(chunk_id)

        self.assertEqual(returned_doc, expected_doc)
        self.mock_helper.search.assert_called_once_with(chunk_id)

    def test_search_document_invalid_input(self):
        with self.assertRaises(ValueError):
            self.docstore.search(-1)

    def test_get_all_chunk_id(self):
        expected_ids = [1, 2, 3]
        self.mock_helper.get_all_chunk_id.return_value = expected_ids

        returned_ids = self.docstore.get_all_chunk_id()

        self.assertEqual(returned_ids, expected_ids)
        self.mock_helper.get_all_chunk_id.assert_called_once()

    def test_init_invalid_params(self):
        with self.assertRaises(ValueError):  # Invalid URL - string, not URL object
            OpenGaussDocstore("not a engine")

        with self.assertRaises(ValueError):
            OpenGaussDocstore(
                self.mock_engine, encrypt_fn=123
            )  # Invalid encrypt_fn

        with self.assertRaises(ValueError):
            OpenGaussDocstore(
                self.mock_engine, decrypt_fn=123
            )  # Invalid decrypt_fn

    def test_full_text_search(self):
        res = self.docstore.full_text_search("test")  # 数据库底层行为无法模拟，暂时预留此用例
        self.assertEqual(res, [])
        self.docstore.drop()

    def test_search_by_document_id(self):
        self.mock_helper.search_by_document_id.return_value = self.test_documents
        res = self.docstore.search_by_document_id(1)
        self.assertEqual(res[0].page_content, "content1")
        self.mock_helper.search_by_document_id.assert_called_once()

    def test_update(self):
        self.mock_helper.update.return_value = None
        self.docstore.update([1, 2], ["test1", "test2"])
        self.mock_helper.update.assert_called_once()

    def test_fake_engine(self):
        mock_engine = MagicMock(spec=Engine)
        mock_engine.name = "mysql"
        with self.assertRaises(StorageError):
            OpenGaussDocstore(engine=mock_engine, enable_bm25=False)


if __name__ == "__main__":
    unittest.main()
