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
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from transformers import is_torch_npu_available

from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.knowledge import KnowledgeDB
from langchain_core.documents import Document
from mx_rag.storage.document_store import MxDocument

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.retrievers import Retriever
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore

EMBEDDING_TEXT = """The unshare command creates new namespaces and then executes the specified program."""


class MyTestCase(unittest.TestCase):
    sql_db_file = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    def test_Retriever_npu(self):
        if not is_torch_npu_available():
            logger.info("skip npu case")
            return

        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore(MyTestCase.sql_db_file)
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0],
                          load_local_index="./faiss.index")
        knowledge_store = KnowledgeStore(MyTestCase.sql_db_file)
        knowledge_store.add_knowledge(knowledge_name='test', user_id='Default')
        knowledge_db = KnowledgeDB(knowledge_store, db, index, "test", white_paths=["/home"], user_id='Default')
        knowledge_db.add_file("unshare_desc.txt", [EMBEDDING_TEXT], embed_func=emb.embed_documents)
        logger.info("create MindFAISS done")
        r = Retriever(vector_store=index, document_store=db, score_threshold=0.5, embed_func=emb.embed_documents)

        def test_result(self):
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_result_with_prompt(self):
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        def test_no_result(self):
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(query, docs[0].page_content)

        def test_no_result_with_prompt(self):
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(query, docs[0].page_content)

        test_result(self)
        test_result_with_prompt(self)
        test_no_result(self)
        test_no_result_with_prompt(self)

    def test_Retriever(self):
        if is_torch_npu_available():
            logger.info("skip none npu case")
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        shutil.disk_usage = MagicMock(return_value=(1, 1, 1000 * 1024 * 1024))
        db = SQLiteDocstore(MyTestCase.sql_db_file)
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0], load_local_index="./faiss.index")

        r = Retriever(vector_store=vector_store, document_store=db, score_threshold=0.5, embed_func=embed_func)

        def mock_vector_store_search(embeddings: np.ndarray, k: int = 3, filter_dict=None):
            return [[0.6]], [[0]]

        @patch("mx_rag.storage.vectorstore.faiss_npu.MindFAISS.search", side_effect=mock_vector_store_search)
        # @patch("mx_rag.storage.document_store.SQLiteDocstore.search")
        def test_result(self, mock_vector_store_search):
            db.search = MagicMock(
                return_value=MxDocument(page_content="this is test", metadata={},
                                        document_name="mindie.docx"))
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            self.assertEqual("this is test", docs[0].page_content)

        @patch("mx_rag.retrievers.retriever.Retriever._get_relevant_documents")
        def test_result_with_prompt(self, get_relevant_documents_mock):
            get_relevant_documents_mock.return_value = [Document(page_content=EMBEDDING_TEXT, metadata={})]
            prompt = "haha"
            query = "what is unshare command?"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        @patch("mx_rag.retrievers.retriever.Retriever._get_relevant_documents")
        def test_no_result(self, get_relevant_documents_mock):
            get_relevant_documents_mock.return_value = [Document(page_content=EMBEDDING_TEXT, metadata={})]
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)
            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        @patch("mx_rag.retrievers.retriever.Retriever._get_relevant_documents")
        def test_no_result_with_prompt(self, get_relevant_documents_mock):
            get_relevant_documents_mock.return_value = [Document(page_content=EMBEDDING_TEXT, metadata={})]
            prompt = "haha"
            query = "xxxx xxx xx xxx xxx x"
            logger.info(f"get_relevant_documents ['{query}']")
            docs = r.invoke(query)

            logger.info(f"relevant doc {docs}")
            self.assertEqual(EMBEDDING_TEXT, docs[0].page_content)

        test_result(self)
        test_result_with_prompt(self)
        test_no_result(self)
        test_no_result_with_prompt(self)


if __name__ == '__main__':
    unittest.main()
