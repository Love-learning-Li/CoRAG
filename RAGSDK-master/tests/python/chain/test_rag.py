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
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document
from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.chain import SingleText2TextChain
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever, MultiQueryRetriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS


class MyTestCase(unittest.TestCase):
    sql_db_file = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

        os.makedirs("./bge-large-zh/")

    def tearDown(self) -> None:
        shutil.rmtree("./bge-large-zh/")

    def test_with_npu(self):
        if not is_torch_npu_available():
            return

        emb = TextEmbedding("./bge-large-zh/")
        db = SQLiteDocstore("./sql.db")
        logger.info("create emb done")
        logger.info("set_device done")
        os.system = MagicMock(return_value=0)
        index = MindFAISS(x_dim=1024, devs=[0],
                          load_local_index="./faiss.index")
        knowledge_store = KnowledgeStore("./sql.db")
        knowledge_store.add_knowledge(knowledge_name='test', user_id='Default')
        vector_store = KnowledgeDB(knowledge_store, db, index, "test", white_paths=["/home"],
                                   user_id='Default')
        vector_store.add_file("test_file.txt", ["this is a test"], metadatas=[{"filepath": "xxx.file"}],
                              embed_func=emb.embed_documents)
        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", base_url="http://71.14.88.12:7890")

        def test_rag_chain_npu(self):
            r = Retriever(vector_store=vector_store, document_store=db, embed_func=emb.embed_documents)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            response = rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
            logger.debug(f"response {response}")

        def test_rag_chain_npu_stream(self):
            r = Retriever(vector_store=vector_store, document_store=db, embed_func=emb.embed_documents)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            for response in rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0,
                                                                          top_p=0.1, stream=True)):
                logger.debug(f"stream response {response}")

        def test_rag_chain_npu_multi_query_retriever(self):
            r = MultiQueryRetriever(llm=llm, vector_store=vector_store, embed_func=emb.embed_documents)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            response = rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
            logger.debug(f"response {response}")

        def test_rag_chain_npu_stream_multi_query_retriever(self):
            r = MultiQueryRetriever(llm=llm, vector_store=vector_store, embed_func=emb.embed_documents)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            rag.source = True
            for response in rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0,
                                                                          top_p=0.1, stream=True)):
                logger.debug(f"stream response {response}")

        test_rag_chain_npu(self)
        test_rag_chain_npu_stream(self)
        test_rag_chain_npu_multi_query_retriever(self)
        test_rag_chain_npu_stream_multi_query_retriever(self)

    def test_with_no_npu(self):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024)).tolist()

        shutil.disk_usage = MagicMock(return_value=(1, 1, 1000 * 1024 * 1024))
        db = SQLiteDocstore("sql.db")
        os.system = MagicMock(return_value=0)
        vector_store = MindFAISS(x_dim=1024, devs=[0],
                                 load_local_index="./faiss.index")
        vector_store.similarity_search = MagicMock(
            return_value=[[(Document(page_content="this is a test", document_name="test.txt"), 0.5)]])
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", base_url="http://127.0.0.1:7890")

        @patch("mx_rag.llm.Text2TextLLM.chat")
        def test_rag_chain_npu(self, chat_mock):
            r = Retriever(vector_store=vector_store, document_store=db, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            chat_mock.return_value = "test test test"
            response = rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
            self.assertEqual("test test test", response.get("result"))

        @patch("mx_rag.llm.Text2TextLLM.chat_streamly")
        def test_rag_chain_npu_stream(self, chat_mock):
            r = Retriever(vector_store=vector_store, document_store=db, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            chat_mock.return_value = (yield "Retriever steam")
            for response in rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0,
                                                                          top_p=0.1, stream=True)):
                self.assertEqual("Retriever steam", response.get("result"))

        @patch("mx_rag.llm.Text2TextLLM.chat")
        def test_rag_chain_npu_multi_query_retriever(self, chat_mock):
            r = MultiQueryRetriever(llm=llm, vector_store=vector_store, document_store=db, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            chat_mock.return_value = ("MultiQueryRetriever")
            response = rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
            self.assertEqual("MultiQueryRetriever", response.get("result"))

        @patch("mx_rag.llm.Text2TextLLM.chat_streamly")
        def test_rag_chain_npu_stream_multi_query_retriever(self, chat_mock):
            r = MultiQueryRetriever(llm=llm, vector_store=vector_store, document_store=db, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            rag.source = True
            chat_mock.return_value = (yield "MultiQueryRetriever steam")
            for response in rag.query("who are you??", LLMParameterConfig(max_tokens=1024, temperature=1.0,
                                                                          top_p=0.1, stream=True)):
                self.assertEqual("MultiQueryRetriever steam", response.get("result"))

        test_rag_chain_npu(self)
        test_rag_chain_npu_stream(self)
        test_rag_chain_npu_multi_query_retriever(self)
        test_rag_chain_npu_stream_multi_query_retriever(self)


if __name__ == '__main__':
    unittest.main()
