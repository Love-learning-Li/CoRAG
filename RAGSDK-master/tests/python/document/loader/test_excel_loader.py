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
from unittest import mock
from unittest.mock import patch, MagicMock

from mx_rag.document.loader.excel_loader import ExcelLoader
from mx_rag.document.loader.base_loader import BaseLoader

class TestExcelLoader(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_class_init_case(self):
        loader = ExcelLoader(os.path.join(self.data_dir, "test.xlsx"))
        self.assertIsInstance(loader, ExcelLoader)

    @patch.object(ExcelLoader, '_load_xls')
    def test_call_load_xls(self, mock_load_xls):
        # Arrange
        loader = ExcelLoader(os.path.join(self.data_dir, "test.xls"))
        loader.lazy_load()
        mock_load_xls.assert_called_once()

    def test_load_xls(self):
        loader = ExcelLoader(os.path.join(self.data_dir, "test.xls"))
        docs = loader.load()
        contents = [
            ':中文资讯;source:测试1;link:www.test1.com;count(3月15日为例):9篇;SUM:24篇;',
            ':中文资讯;source:测试3;link:www.test3.com;count(3月15日为例):4篇;SUM:24篇;',
            ':中文资讯;source:测试4;link:www.test4.com;count(3月15日为例):4篇;SUM:24篇;',
            ':中文资讯;source:测试5;link:行业资讯 | 公园 (test.net);count(3月15日为例):7篇;SUM:24篇;',
            ':英文文献;source:测试6;link:行业资讯 | 公园 (test.net);count(3月15日为例):14篇;SUM:55篇;',
            ':英文文献;source:测试7;link:行业资讯 | 公园 (test.net);count(3月15日为例):41篇;SUM:55篇;']
        self.assertEqual(len(contents), len(docs))
        self.assertEqual(contents[0], docs[0].page_content)
        self.assertEqual(docs[0].metadata["sheet"], '不需要订阅')

    @patch.object(ExcelLoader, '_load_xlsx')
    def test_call_load_xlsx(self, mock_load_xlsx):
        # Arrange
        loader = ExcelLoader(os.path.join(self.data_dir, "test.xlsx"))
        loader.lazy_load()
        mock_load_xlsx.assert_called_once()

    def test_load_xlsx(self):
        loader = ExcelLoader(os.path.join(self.data_dir, "test.xlsx"))
        docs = loader.load()
        contents = [
            ':中文资讯;source:test1;link:www.test1.com;count(3月15日为例):9篇;SUM:24篇;',
            ':中文资讯;source:test3;link:www.test3.com;count(3月15日为例):4篇;SUM:24篇;',
            ':中文资讯;source:test4;link:www.test4.com;count(3月15日为例):4篇;SUM:24篇;',
            ':中文资讯;source:test5;link:www.test5.com;count(3月15日为例):7篇;SUM:24篇;',
            ':英文文献;source:test6;link:www.test6.com;count(3月15日为例):14篇;SUM:55篇;',
            ':英文文献;source:test7;link:www.test7.com;count(3月15日为例):41篇;SUM:55篇;']
        self.assertEqual(len(docs), len(contents))
        for idx, content in enumerate(contents):
            self.assertEqual(docs[idx].page_content, content)
        self.assertEqual(docs[0].metadata["sheet"], '不需要订阅')
        with patch("mx_rag.document.loader.base_loader.BaseLoader.MAX_PAGE_NUM", new=1):
            loader = ExcelLoader(os.path.join(self.data_dir, "test.xlsx"))
            self.assertEqual(loader.load(), [])

    def test_load_csv(self):
        loader = ExcelLoader(os.path.join(self.data_dir, "test.csv"))
        with self.assertRaises(ValueError):
            docs = loader.load()


if __name__ == '__main__':
    unittest.main()
