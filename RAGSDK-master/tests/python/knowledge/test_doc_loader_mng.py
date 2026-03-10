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
import unittest
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document import LoaderMng
from mx_rag.document.doc_loader_mng import LoaderInfo, SplitterInfo
from mx_rag.document.loader import ExcelLoader


class LoaderMngTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../tests/data/test.xlsx"))

    def test_register_loader(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_info = loader_mng.get_loader(".xlsx")
        loader = loader_info.loader_class(file_path=self.data_dir)
        self.assertIsInstance(loader, ExcelLoader)

    def test_register_splitter(self):
        loader_mng = LoaderMng()
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".docx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        splitter_info = loader_mng.get_splitter(".xlsx")
        splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)
        with self.assertRaises(ValueError):
            loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                         {"chunk_size": 4000, 'test': {"chunk_overlap": {"keep_separator": False}}})
        with self.assertRaises(KeyError):
            loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".jpg", ".png"],
                                         {"chunk_size": 4000})

    def test_unregister_loader(self):
        loader_mng = LoaderMng()
        with self.assertRaises(KeyError):
            loader_mng.unregister_loader(ExcelLoader)
        loader_mng.register_loader(ExcelLoader, [".xlsx", ".docx", ".doc"])
        loader_mng.unregister_loader(ExcelLoader, ".docx")
        with self.assertRaises(KeyError):
            loader_mng.unregister_loader(ExcelLoader, ".docxx")
        loader_mng.unregister_loader(ExcelLoader)
        with self.assertRaises(KeyError):
            loader_mng.get_loader(".xlsx")

    def test_unregister_splitter(self):
        loader_mng = LoaderMng()
        with self.assertRaises(KeyError):
            loader_mng.unregister_splitter(RecursiveCharacterTextSplitter, '.docx')
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        loader_mng.unregister_splitter(RecursiveCharacterTextSplitter, '.docx')
        with self.assertRaises(KeyError):
            loader_mng.unregister_splitter(RecursiveCharacterTextSplitter, '.docxx')

        loader_mng.unregister_splitter(RecursiveCharacterTextSplitter)
        with self.assertRaises(KeyError):
            loader_mng.get_splitter(".xlsx")

    def test_get_loader(self):
        loader_mng = LoaderMng()
        loader_mng.register_loader(ExcelLoader, [".xlsx"])
        loader_info = loader_mng.get_loader(".xlsx")
        self.assertIsInstance(loader_info, LoaderInfo)

    def test_get_splitter(self):
        loader_mng = LoaderMng()
        loader_mng.register_splitter(RecursiveCharacterTextSplitter, [".xlsx", ".docx", ".doc", ".pdf", ".pptx"],
                                     {"chunk_size": 4000, "chunk_overlap": 20, "keep_separator": False})
        splitter_info = loader_mng.get_splitter(".xlsx")
        self.assertIsInstance(splitter_info, SplitterInfo)

if __name__ == '__main__':
    unittest.main()