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
from unittest.mock import patch

import numpy as np

from mx_rag.tools.finetune.dataprocess.reranker_featured import reranker_featured
from mx_rag.reranker.local import LocalReranker


class TestRerankerFeatured(unittest.TestCase):

    @patch("mx_rag.reranker.local.LocalReranker.__init__")
    @patch("mx_rag.reranker.local.LocalReranker.rerank")
    def test_run_success(self, fake_rerank, fake_init):
        def f_reranker(query: str, texts: list[str]):
            return np.array([1] * len(texts))

        fake_init.return_value = None
        fake_rerank.side_effect = f_reranker
        reranker = LocalReranker("test_rerank_name")
        scores = reranker_featured(reranker, ["query"], ["doc"])
        self.assertEqual(scores, [1])


if __name__ == '__main__':
    unittest.main()
