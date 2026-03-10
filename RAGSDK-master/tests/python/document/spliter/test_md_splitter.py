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

from mx_rag.document.splitter import MarkdownTextSplitter


class MarkdownTextSplitterTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def setUp(self):
        """Setup test fixtures"""
        self.sample_md = os.path.join(self.data_dir, "example.md")
        with open(self.sample_md, "r") as f:
            self.sample_content = f.read()

    def test_basic_splitting(self):
        """Test basic text splitting functionality"""
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(self.sample_content)
        self.assertGreater(len(chunks), 1)


    def test_header_based_splitting(self):
        """Test splitting by markdown headers"""
        splitter = MarkdownTextSplitter(chunk_size=1000, header_level=3)
        chunks = splitter.split_text(self.sample_content)
        self.assertTrue(any("### 三级标题" in chunk for chunk in chunks))

    def test_small_chunk_merging(self):
        """Test merging of small chunks"""
        splitter = MarkdownTextSplitter(chunk_size=2000, header_level=2)
        chunks = splitter.split_text(self.sample_content)
        self.assertLess(len(chunks), 5)

    def test_header_metadata_recovery(self):
        """Test recovery of header metadata in chunks"""
        splitter = MarkdownTextSplitter(header_level=2)
        chunks = splitter.split_text(self.sample_content)
        first_chunk = chunks[0]
        self.assertIn("# 一级标题", first_chunk)

    def test_large_content_recursive_split(self):
        """Test recursive splitting of large content"""
        large_text = "# Header\n" + "Content " * 1000
        splitter = MarkdownTextSplitter(chunk_size=500)
        chunks = splitter.split_text(large_text)
        self.assertGreater(len(chunks), 1)

    def test_invalid_header_level(self):
        """Test handling of invalid header levels"""
        with self.assertRaises(ValueError):
            MarkdownTextSplitter(header_level=7)

    def test_empty_input(self):
        """Test handling of empty input"""
        splitter = MarkdownTextSplitter()
        chunks = splitter.split_text("")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "")

    def test_metadata_merge(self):
        """Test proper merging of metadata"""
        splitter = MarkdownTextSplitter(header_level=3)
        test_text = "# Main\nContent1\n## Sub\nContent2"
        chunks = splitter.split_text(test_text)
        self.assertEqual(len(chunks), 1)


if __name__ == '__main__':
    unittest.main()