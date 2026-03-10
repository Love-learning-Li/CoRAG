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

import numpy as np
from loguru import logger
from langchain_core.documents import Document
from mx_rag.utils.common import validate_sequence, validate_list_list_str, validate_list_document, check_header, \
    check_embed_func, _check_sparse_and_dense


class TestCommon(unittest.TestCase):
    def test_validate_str(self):
        # 字符串长度超过规格
        data = "a" * 1025
        self.assertFalse(validate_sequence(data))

    def test_validate_list(self):
        # 列表中元素长度超过规格
        data = ["a" * 1025]
        self.assertFalse(validate_sequence(data))

        # 列表长度超过规格
        data = ["a"] * 1025
        self.assertFalse(validate_sequence(data))

        data = [["a"]]
        self.assertFalse(validate_sequence(data))

    def test_validate_tuple(self):
        # 元组中元素长度超过规格
        data = ("a" * 1025,)
        self.assertFalse(validate_sequence(data))

        # 元组长度超过规格
        data = ("a",) * 1025
        self.assertFalse(validate_sequence(data))

    def test_validate_dict(self):
        # 字典中key长度超过规格
        data = {"a" * 1025: 1}
        self.assertFalse(validate_sequence(data))

        # 字典中value长度超过规格
        data = {"a": "b" * 1025}
        self.assertFalse(validate_sequence(data))

    def test_validate_dict_list(self):
        log_file = "./test.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(log_file, format="{message}", level="DEBUG")
        # 层数超过1错误
        validate_sequence({"a": ["b"]})
        # 第二层长度超过规格
        validate_sequence({"a": ["b"] * 1025}, max_check_depth=2)
        # 第三层长度超过规格错误
        validate_sequence({"a": [["b"] * 1025]}, max_check_depth=3)
        with open("./test.log") as fd:
            res1 = fd.readline()
            self.assertTrue(res1.find("nested depth cannot exceed 1") > -1)
            res2 = fd.readline()
            self.assertTrue(res2.find("1th layer param length") > -1)
            res3 = fd.readline()
            self.assertTrue(res3.find("2th layer param length") > -1)
            fd.close()

    def test_validate_list_list_str(self):
        res = validate_list_list_str('x', [1, 10], [1, 10], [1, 10])
        self.assertFalse(res)

        res = validate_list_list_str(['x'], [1, 2, 10], [1, 10], [1, 10])
        self.assertFalse(res)

        res = validate_list_list_str(['x'] * 11, [1, 10], [1, 10], [1, 10])
        self.assertFalse(res)

        res = validate_list_list_str([['x' * 11]], [1, 10], [1, 10], [1, 10])
        self.assertFalse(res)

        res = validate_list_list_str([['x']], [1, 10], [1, 10], [1, 10])
        self.assertTrue(res)

    def test_validate_list_document(self):
        self.assertFalse(validate_list_document('x', [1, 10], [1, 10]))
        self.assertFalse(validate_list_document(['x'] * 11, [1, 10], [1, 10]))
        self.assertFalse(validate_list_document(['x'], [1, 10], [1, 10]))

        data1 = Document(page_content='one_text' * 10, metadata={"source": 'file_path'})
        self.assertFalse(validate_list_document([data1], [1, 10], [1, 10]))

        data = Document(page_content='one_text', metadata={"source": 'file_path'})
        self.assertTrue(validate_list_document([data], [1, 10], [1, 10]))

    def test_check_header(self):
        self.assertFalse(check_header('test'))
        original_dict = {'test': 1}
        new_dict = {f'{key}_{i}': original_dict[key] for i in range(101) for key in original_dict}
        self.assertFalse((check_header(original_dict)))
        self.assertFalse(check_header(new_dict))
        self.assertFalse(check_header({'test' * 50: '1'}))
        self.assertFalse(check_header({'test': '1\n'}))
        self.assertTrue(check_header({'test': '1'}))

    def test_check_embed_func(self):
        def test_fun():
            return "test"

        self.assertTrue(check_embed_func(test_fun))
        self.assertTrue(check_embed_func({'dense': test_fun, 'sparse': None}))
        self.assertTrue(check_embed_func({'dense': None, 'sparse': test_fun}))
        self.assertTrue(check_embed_func({'dense': test_fun, 'sparse': test_fun}))
        self.assertFalse(check_embed_func({'dense': None, 'sparse': None}))
        self.assertFalse(check_embed_func({'dense': None, 'ss': test_fun}))
        self.assertFalse(check_embed_func({'dense': None}))

    def test_check_sparse_and_dense(self):
        dense = np.array([[0.1, 0.2], [0.3, 0.4]])
        sparse = [{1: 0.1, 2: 0.2}, {3: 0.3, 4: 0.4}]
        _check_sparse_and_dense([1, 2], dense, sparse)
        with self.assertRaises(ValueError):
            _check_sparse_and_dense([1, 2, 3], dense, sparse)
