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
import time
import unittest
from unittest import mock
from unittest.mock import patch

from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.chain import ParallelText2TextChain
from mx_rag.storage.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore


class TestParallelChain(unittest.TestCase):
    sql_db_file = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/sql.db"))

    def setUp(self):
        if os.path.exists(TestParallelChain.sql_db_file):
            os.remove(TestParallelChain.sql_db_file)

    def test_init(self):
        db = SQLiteDocstore(TestParallelChain.sql_db_file)
        vector_store = MindFAISS(x_dim=1024, devs=[0],
                                 load_local_index="./faiss.index")
        retrieve = Retriever(vector_store=vector_store, document_store=db,
                             embed_func=lambda input_list: [[float(num) for num in sub.split()] for sub in input_list],
                             k=1, score_threshold=0.1)
        llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", base_url="http://70.255.71.175:3000", timeout=120)
        parallel_chain = ParallelText2TextChain(llm=llm, retriever=retrieve)
        self.assertIsInstance(parallel_chain, ParallelText2TextChain)

    def test_query_prefill_first_done(self):
        db = SQLiteDocstore(TestParallelChain.sql_db_file)
        vector_store = MindFAISS(x_dim=1024, devs=[0],
                                 load_local_index="./faiss.index")
        retrieve = Retriever(vector_store=vector_store, document_store=db,
                             embed_func=lambda input_list: [[float(num) for num in sub.split()] for sub in input_list],
                             k=1, score_threshold=0.1)
        llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", base_url="http://70.255.71.175:3000", timeout=120)

        def mock_do_stream_query(*args, **kwargs):
            yield "prefill query done"

        def mock_retrieve_process(*args, **kwargs):
            time.sleep(0.2)

        with patch('mx_rag.chain.SingleText2TextChain._do_stream_query',
                   mock.Mock(side_effect=mock_do_stream_query)):
            with patch('mx_rag.chain.ParallelText2TextChain._retrieve_process',
                       mock.Mock(side_effect=mock_retrieve_process)):
                parallel_chain = ParallelText2TextChain(llm=llm, retriever=retrieve)
                answer = parallel_chain.query(text="123456")
                self.assertEqual(answer, "prefill query done")
                self.assertEqual(parallel_chain.prefill_done.value, 0)

    def test_query_retrieve_first_done(self):
        db = SQLiteDocstore(TestParallelChain.sql_db_file)
        vector_store = MindFAISS(x_dim=1024, devs=[0],
                                 load_local_index="./faiss.index")
        retrieve = Retriever(vector_store=vector_store, document_store=db,
                             embed_func=lambda input_list: [[float(num) for num in sub.split()] for sub in input_list],
                             k=1, score_threshold=0.1)
        llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct", base_url="http://70.255.71.175:3000", timeout=120)

        def mock_do_stream_query(*args, **kwargs):
            time.sleep(0.2)
            yield "prefill query done"

        def mock_retrieve_process(*args, **kwargs):
            pass

        def mock_do_query(*args, **kwargs):
            return "retrieve query done"

        with patch('mx_rag.chain.SingleText2TextChain._do_stream_query',
                   mock.Mock(side_effect=mock_do_stream_query)):
            with patch('mx_rag.chain.ParallelText2TextChain._retrieve_process',
                       mock.Mock(side_effect=mock_retrieve_process)):
                with patch('mx_rag.chain.SingleText2TextChain._do_query',
                           mock.Mock(side_effect=mock_do_query)):
                    parallel_chain = ParallelText2TextChain(llm=llm, retriever=retrieve)
                    answer = parallel_chain.query("123456")
                    self.assertEqual(answer, "retrieve query done")
                    self.assertEqual(parallel_chain.prefill_done.value, 0)


if __name__ == '__main__':
    unittest.main()
