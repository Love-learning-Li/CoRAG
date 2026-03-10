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

import os.path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from mx_rag.storage.vectorstore import MindFAISS, VectorStore
from mx_rag.storage.vectorstore.faiss_npu import MindFAISSError


class TestMindFAISS(unittest.TestCase):
    def test_faiss(self):
        with patch("mx_rag.storage.vectorstore.faiss_npu.ascendfaiss") as ascendfaiss:
            with patch("mx_rag.storage.vectorstore.faiss_npu.faiss") as faiss:

                total = np.random.random((3, 1024))
                query = np.array([total[0]])

                os.system = MagicMock(return_value=0)
                os.chmod = MagicMock()

                with self.assertRaises(KeyError):
                    index = MindFAISS.create(devs=[0],
                                             load_local_index="./faiss.index")
                with self.assertRaises(KeyError):
                    index = MindFAISS.create(x_dim=1024,
                                             load_local_index="./faiss.index")
                with self.assertRaises(KeyError):
                    index = MindFAISS.create(x_dim=1024, devs=[0])
                with self.assertRaises(MindFAISSError):
                    index = MindFAISS(x_dim=1024, devs=0,
                                      load_local_index="./faiss.index")
                with self.assertRaises(MindFAISSError):
                    index = MindFAISS.create(x_dim=1024, devs=[0, 1],
                                             load_local_index="./faiss.index")

                index = MindFAISS.create(x_dim=1024, devs=[0],
                                         load_local_index="./faiss.index")

                index.search(query.tolist(), k=1)
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(3, 2, 1024)
                    index.search(vecs.tolist())

                index.add([1], query)
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(3, 2, 1024)
                    index.add([0, 1, 2], vecs)
                with self.assertRaises(MindFAISSError):
                    vecs = np.random.randn(2, 1024)
                    index.add([0, 1, 2], vecs)
                with patch.object(VectorStore, 'MAX_VEC_NUM', 1):
                    with self.assertRaises(MindFAISSError):
                        vecs = np.random.randn(3, 1024)
                        index.add([0, 1, 2], vecs)
                    with self.assertRaises(MindFAISSError):
                        index.delete([1, 2, 3])
                with patch.object(VectorStore, 'MAX_SEARCH_BATCH', 1):
                    with self.assertRaises(MindFAISSError):
                        vecs = np.random.randn(3, 1024)
                        index.search(vecs.tolist())
                with self.assertRaises(AttributeError):
                    index.get_all_ids()
                index.delete([1])
                index.save_local()
                index.get_save_file()
