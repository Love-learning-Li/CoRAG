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

import numpy as np


def write_index(*args, **kwargs):
    return ""


def read_index(*args, **kwargs):
    return ""


METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1


class IndexFlatIP:
    def __init__(self, embed_len: int):
        self.embed_len = embed_len

    def add(self, embedding: list):
        pass

    def search(self, batch_embedding: list, k: int):
        return np.array([i for i in range(len(batch_embedding))]), \
            np.array([[i for i in range(k)]] * len(batch_embedding))


class IndexHNSWFlat:
    def __init__(self, embed_len: int, m: int = 16):
        self.embed_len = embed_len
        self.m = m

    def add(self, embedding: list):
        pass

    def search(self, batch_embedding: list, k: int):
        return np.array([i for i in range(len(batch_embedding))]), \
            np.array([[i for i in range(k)]] * len(batch_embedding))
