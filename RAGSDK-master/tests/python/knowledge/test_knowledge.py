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
import pathlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.base_knowledge import KnowledgeError
from mx_rag.knowledge.knowledge import KnowledgeStore, _check_embedding
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS

SQL_PATH = "./sql.db"


class TestKnowledge(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.md"))

    def setUp(self):
        # 先清空临时数据库
        if os.path.exists(SQL_PATH):
            os.remove(SQL_PATH)

    @patch("mx_rag.knowledge.KnowledgeDB._check_store_accordance")
    def test_knowledge(self, knowledge_db_mock):
        embeddings = np.concatenate([np.random.random((1, 1024))])

        def embed_func(texts):
            return embeddings.tolist()

        db = SQLiteDocstore(SQL_PATH)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        top_path = os.path.dirname(os.path.dirname(current_dir))
        vector_store = MagicMock(spec=MindFAISS)
        vector_store.add = MagicMock(return_value=None)
        knowledge_store = KnowledgeStore(SQL_PATH)
        knowledge_store.add_knowledge("test_knowledge", "user123", "admin")
        knowledge_store.add_knowledge("test_knowledge", "user123", 'admin')
        self.assertEqual(knowledge_store.check_knowledge_exist("test_knowledge", "user123"), True)
        knowledge_store.add_usr_id_to_knowledge("test_knowledge", "user124", "admin")
        knowledge_store.add_usr_id_to_knowledge("test_knowledge", "user124", "admin")
        knowledge_store.add_usr_id_to_knowledge("test_knowledge", "user000", "member")
        knowledge_store.add_doc_info("test_knowledge", "doc_name", "file_path", "user123")
        with self.assertRaises(KnowledgeError):
            knowledge_store.add_doc_info("test_knowledge1", "doc_name", "file_path", "user123")

        with self.assertRaises(KnowledgeError):
            knowledge_store.add_doc_info("test_knowledge", "doc_name", "file_path", "user000")

        with self.assertRaises(KnowledgeError):
            knowledge_store.add_usr_id_to_knowledge("test_knowledge001", "user124", "admin")

        knowledge = KnowledgeDB(knowledge_store, db, vector_store, "test_knowledge",
                                white_paths=[top_path, ], user_id='user123')
        knowledge.add_file(pathlib.Path(self.test_file), ["this is a test"], metadatas=[{"filepath": "xxx.file"}],
                           embed_func={"dense": embed_func, "sparse": None})
        with self.assertRaises(KnowledgeError):
            knowledge.add_file(pathlib.Path(self.test_file), ["this is a test"],
                               metadatas=[{"filepath": "xxx.file"}, {"filepath": "yyy.file"}],
                               embed_func={"dense": embed_func, "sparse": None})

        self.assertEqual(knowledge.get_all_documents()[0].knowledge_name, "test_knowledge")
        self.assertEqual(knowledge.get_all_documents()[0].document_name, "doc_name")

        knowledge_db_mock.return_value = None
        knowledge_db1 = KnowledgeDB(KnowledgeStore(SQL_PATH), db, vector_store, "test_knowledge",
                                    white_paths=[top_path, ], user_id="user123")
        self.assertEqual(knowledge_db1.get_all_documents()[0].knowledge_name, "test_knowledge")
        self.assertEqual(knowledge_db1.get_all_documents()[0].document_name, "doc_name")
        self.assertEqual(knowledge_store._get_all_knowledge_name('user123'), ["test_knowledge"])

        # 删除文档后, 只剩下空的knowledge
        knowledge.delete_file("test.md")
        self.assertEqual(knowledge_store._get_all_knowledge_name('user123'), ["test_knowledge"])
        self.assertEqual(knowledge_store.get_all_usr_role_by_knowledge("test_knowledge"),
                         {'user000': 'member', 'user123': 'admin', 'user124': 'admin'})
        self.assertEqual(knowledge_store.get_all_usr_role_by_knowledge("test_knowledge001"), {})
        # 多个usr_id对knowledge关系删除
        knowledge_store.delete_usr_id_from_knowledge("test_knowledge", "user123", 'admin')
        # user_id和knowledge1对1时，不允许删除关系，使用delete_knowledge删除
        with self.assertRaises(KnowledgeError):
            knowledge_store.delete_usr_id_from_knowledge("test_knowledge", "user123", 'admin')
        with self.assertRaises(KnowledgeError):
            knowledge_store.delete_usr_id_from_knowledge("test_knowledge001", "user123", 'admin')

        knowledge_db1.delete_all()
        self.assertEqual(knowledge_store._get_all_knowledge_name('user123'), [])

    def test_check_embedding(self):
        dense = [[0.1, 0.2], [0.3, 0.4]]
        sparse = [{1: 0.1, 2: 0.2}, {3: 0.3, 4: 0.4}]
        _check_embedding("dense", dense, ["text1", "text2"], ["text1", "text2"])
        _check_embedding("sparse", sparse, ["text1", "text2"], ["text1", "text2"])
        with self.assertRaises(KnowledgeError):
            _check_embedding("dense", dense, ["text1"], ["text1", "text2"])


if __name__ == '__main__':
    unittest.main()
