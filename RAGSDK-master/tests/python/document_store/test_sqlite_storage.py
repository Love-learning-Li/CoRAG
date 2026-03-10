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
import os
from pathlib import Path
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.document_store import MxDocument

SQL_PATH = str(Path(__file__).parent.absolute() / "sql.db")


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)
        self.db = SQLiteDocstore(SQL_PATH)

    def tearDown(self):
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    def test_sqlite_storage_add(self):
        # 对add函数入参进行校验测试
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        with self.assertRaises(ValueError):
            # 期望传入一个列表
            self.db.add(doc, 1)
            # 期望列表元素的类型为MxDocument
            self.db.add([0], 1)
        self.assertEqual(self.db.add([doc], 1), [1])

    def test_sqlite_storage_delete(self):
        # 不删除任何chunk，返回空列表
        self.assertEqual(self.db.delete(document_id=1), [])

    def test_sqlite_storage_search(self):
        # 对search函数入参进行校验测试
        with self.assertRaises(ValueError):
            # 期望传入一个整数
            self.db.search(-1)
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        self.db.add([doc], 1)
        chunk = self.db.search(1)
        self.assertEqual(chunk.page_content, "Hello mxRAG")
        self.db.delete(1)
        self.assertEqual(self.db.get_all_chunk_id(), [])

    def test_chunk_encrypt(self):
        def fack_encryt(value):
            return "fack_encryt"

        db = SQLiteDocstore(SQL_PATH, encrypt_fn=fack_encryt)
        doc = MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test")
        db.add([doc], 1)
        chunk = db.search(1)
        self.assertEqual(chunk.page_content, "fack_encryt")

    def test_search_by_document_id(self):
        db = SQLiteDocstore(SQL_PATH)
        doc = [MxDocument(page_content="Hello mxRAG", metadata={"test": "test"}, document_name="test"),
               MxDocument(page_content="Hello mxRAG1", metadata={"test": "test"}, document_name="test"),
               MxDocument(page_content="Hello mxRAG2", metadata={"test": "test"}, document_name="test1"), ]
        db.add(doc, 1)
        q1 = db.search_by_document_id(1)
        self.assertEqual(len(q1), 3)
        q2 = db.search_by_document_id(2)
        self.assertEqual(q2, [])

    def test_update(self):
        db = SQLiteDocstore(SQL_PATH)
        self.test_search_by_document_id()
        db.update([1, 2, 3], ["Hello RAG SDK", "Hello RAG SDK1", "Hello RAG SDK2"])
        docs = db.search_by_document_id(1)
        self.assertEqual(docs[0].page_content, "Hello RAG SDK")
        self.assertEqual(docs[1].page_content, "Hello RAG SDK1")
        self.assertEqual(docs[2].page_content, "Hello RAG SDK2")


if __name__ == '__main__':
    unittest.main()
