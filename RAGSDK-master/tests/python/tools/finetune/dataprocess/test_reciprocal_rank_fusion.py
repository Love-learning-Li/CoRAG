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

from mx_rag.tools.finetune.dataprocess.reciprocal_rank_fusion import reciprocal_rank_fusion


class TestRRF(unittest.TestCase):

    def test_run_success(self):
        a = ["1", "2", "3"]
        b = ["2", "3", "4"]

        c = reciprocal_rank_fusion([a, b])
        self.assertEqual(len(c), 4)


if __name__ == '__main__':
    unittest.main()
