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

from abc import ABC, abstractmethod
from typing import List

from mx_rag.utils.file_check import FileCheck
from mx_rag.utils.common import validate_params, validate_list_str, MAX_PATH_LENGTH


class KnowledgeBase(ABC):
    @validate_params(
        white_paths=dict(validator=lambda x: validate_list_str(x, [1, MAX_PATH_LENGTH], [1, MAX_PATH_LENGTH]),
                         message=f"param must meets: Type is List[str], "
                                 f"list length range [1, {MAX_PATH_LENGTH}], str length range [1, {MAX_PATH_LENGTH}]"))
    def __init__(self, white_paths: List[str]):
        self.white_paths = white_paths
        for white_path in white_paths:
            FileCheck.dir_check(white_path)

    @abstractmethod
    def add_file(self, file, texts, embed_func, metadatas):
        pass

    @abstractmethod
    def check_document_exist(self, doc_name):
        pass

    @abstractmethod
    def delete_file(self, doc_name):
        pass

    @abstractmethod
    def get_all_documents(self):
        pass


class KnowledgeError(Exception):
    pass
