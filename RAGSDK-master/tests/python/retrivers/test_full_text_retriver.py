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
from unittest.mock import MagicMock, patch
from pymilvus import MilvusClient
from langchain_core.documents import Document
from mx_rag.storage.document_store import MilvusDocstore, MxDocument
from mx_rag.retrievers import FullTextRetriever


class TestFullTextRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=MilvusClient)

        # Create instance of MilvusDocStore
        self.store_bm25 = MilvusDocstore(self.mock_client, collection_name="doc_store_bm25")

        self.mock_document = MagicMock(MxDocument)
        self.mock_document.metadata = {"chunk_id": 1, "document_id": 123, "document_name": "Test Document"}
        self.mock_document.page_content = "Some content here."
        self.mock_document.document_name = "Test Document"

    def test_full_text_retriever(self):
        self.mock_client.search.return_value = [[
            {
                "distance": 0.1,
                "entity": {
                    "metadata": {
                        "score": 0.1
                    },
                    "page_content": "Some content here.",
                    "document_name": "test.doc"
                }
            }
        ]]
        full_text_retriever = FullTextRetriever(document_store=self.store_bm25, k=3)
        result = full_text_retriever.invoke("test")
        self.assertIsInstance(result[0], Document)  # Ensure result is an MxDocument instance
        self.assertEqual(result[0].page_content, "Some content here.")  # Check content is correct

if __name__ == '__main__':
    unittest.main()