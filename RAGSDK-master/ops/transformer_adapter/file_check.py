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
from pathlib import Path


class FileCheck:
    MAX_PATH_LENGTH = 1024

    @staticmethod
    def check_file_size(file_path: str, max_file_size: int):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise ValueError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}'... length over limit")

        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise ValueError(f"FileSizeLimit: '{file_path}' size over Limit: {max_file_size}")

    @staticmethod
    def check_file_exist(file_path):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise ValueError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}'... length over limit")

        if not os.path.exists(file_path):
            raise ValueError(f"path '{file_path}' is not exists")

    @staticmethod
    def check_input_path_valid(path: str, check_real_path: bool = True):
        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise ValueError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}'... length over limit")

        if ".." in path:
            raise ValueError(f"there are illegal characters in path '{path}'")

        if check_real_path and Path(path).resolve() != Path(path).absolute():
            raise ValueError(f"Input path '{path}' is not valid")

    @staticmethod
    def check_path_is_exist_and_valid(path: str, check_real_path: bool = True):
        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise ValueError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")

        FileCheck.check_file_exist(path)

        FileCheck.check_input_path_valid(path, check_real_path)

    @staticmethod
    def check_file_owner(file_path: str):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise ValueError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        current_user_uid = os.getuid()

        def check_owner(path: str, path_type: str):
            """辅助函数，用于检查一个文件或目录的属主。"""
            try:
                stat_info = os.stat(path)
                owner_uid = stat_info.st_uid
                if owner_uid != current_user_uid:
                    raise ValueError(f"The owner of the {path_type} '{path}' is different from the current user")
            except FileNotFoundError as fnf_error:
                raise ValueError(f"The {path_type} '{path}' does not exist") from fnf_error
            except PermissionError as pe_error:
                raise ValueError(f"Permission denied when accessing the {path_type} '{path}'") from pe_error

        # 检查文件的属主
        check_owner(file_path, "file")

        # 获取文件所在的目录
        dir_path = os.path.dirname(os.path.abspath(file_path))
        # 检查目录的属主
        check_owner(dir_path, "directory")
