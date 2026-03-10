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

from mx_rag.cache import MxRAGCache
from mx_rag.cache import CacheConfig, SimilarityCacheConfig


class TestCacheCore(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_save_folder = os.path.join(current_dir, "cache_save_folder")

    def clear_cache_file(self):
        if os.path.exists(self.data_save_folder):
            shutil.rmtree(self.data_save_folder)
        if not os.path.exists(self.data_save_folder):
            os.mkdir(self.data_save_folder)

    def test_core_init(self):
        config = CacheConfig(cache_size=100, data_save_folder=self.data_save_folder)
        mxrag_cache = MxRAGCache("test_cache", config)
        self.assertIsInstance(mxrag_cache, MxRAGCache)

    def test_cache_update(self):
        self.clear_cache_file()
        config = CacheConfig(cache_size=100, data_save_folder=self.data_save_folder)

        mxrag_cache = MxRAGCache("test_cache", config)
        mxrag_cache.update("hello world", "yes i am")
        answer = mxrag_cache.search("hello world")
        self.assertEqual(answer, "yes i am")

    def test_cache_join(self):
        self.clear_cache_file()

        config = CacheConfig(cache_size=100, data_save_folder=self.data_save_folder)

        mxrag_l1_cache = MxRAGCache("l1cache", config)
        mxrag_l2_cache = MxRAGCache("l2cache", config)
        mxrag_l1_cache.join(mxrag_l2_cache)

        answer = mxrag_l1_cache.search("hello world")
        self.assertEqual(answer, None)

        mxrag_l1_cache.update("hello world", "i am bob")
        answer = mxrag_l1_cache.search("hello world")
        self.assertEqual(answer, "i am bob")

        answer = mxrag_l2_cache.search("hello world")
        self.assertEqual(answer, "i am bob")

    def test_cache_clear(self):
        self.clear_cache_file()
        config = CacheConfig(cache_size=100, data_save_folder=self.data_save_folder)
        mxrag_cache = MxRAGCache("test_cache", config)
        mxrag_cache.update("hello world", "yes i am")
        answer = mxrag_cache.search("hello world")
        self.assertEqual(answer, "yes i am")
        mxrag_cache.flush()
        file_dir = os.path.dirname(mxrag_cache.data_save_path["txt_file"])
        sql_file = os.path.join(file_dir + "/sql.db")
        import sqlite3
        conn = sqlite3.connect(sql_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            )
        ''')
        conn.commit()
        mxrag_cache.data_save_path["sql_file"] = sql_file
        mxrag_cache.clear()
        mxrag_cache_new = MxRAGCache("test_cache", config)
        answer = mxrag_cache_new.search("hello world")
        self.assertEqual(answer, None)


if __name__ == '__main__':
    unittest.main()
