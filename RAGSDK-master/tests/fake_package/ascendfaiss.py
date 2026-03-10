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


class AscendIndexFlat:
    def __init__(self, *args, **kwargs):
        self.ntotal = 0
        pass

    def add_with_ids(self, *args, **kwargs):
        pass

    def search(self, *args, **kwarg):
        return np.array([[0.1]]), np.array([[0]])

    def remove_ids(self, *args, **kwarg):
        pass


class AscendIndexFlatConfig:
    def __init__(self, *args, **kwargs):
        pass


class IntVector:
    def __init__(self, *args, **kwargs):
        pass

    def push_back(self, *args, **kwargs):
        pass


def index_cpu_to_ascend(*args, **kwargs):
    return AscendIndexFlat()


def index_ascend_to_cpu(*args, **kwargs):
    return ""
