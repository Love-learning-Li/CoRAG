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
from pathlib import Path
from loguru import logger


class SizeOverLimitException(Exception):
    pass


class PathNotFileException(Exception):
    pass


class PathNotDirException(Exception):
    pass


class FileCheckError(Exception):
    pass


class SecFileCheck:
    def __init__(self, file_path, max_size, mode_limit=0o755):
        self.file_path = file_path
        self.max_size = max_size
        self.mode_limit = mode_limit

    def check(self):
        FileCheck.check_path_is_exist_and_valid(self.file_path)

        if not os.path.isfile(self.file_path):
            raise PathNotFileException(f"PathNotFileException: '{self.file_path}' is not file")

        FileCheck.check_file_size(self.file_path, self.max_size)
        FileCheck.check_file_owner(self.file_path)
        FileCheck.check_mode(self.file_path, self.mode_limit)


class SecDirCheck:
    """
    功能描述:
        检查目录下在文件是否满足要求

    parameters:
        dir_path: 目录路径
        max_size: 目录下，包含子目录中的文件size最大值
        mode: 目录下所有文件最大权限
        max_depth：目录下子目录最大深度，目录深度从1开始计数
        max_file_num：目录下所有的文件总数上限
    """

    def __init__(self, dir_path, max_size, mode=0o755, max_depth=64, max_file_num=512):
        self.dir_path = dir_path
        self.max_size = max_size
        self.mode = mode
        self.max_depth = max_depth
        self.max_file_num = max_file_num
        self._cur_file_num = 0

    def check(self):
        self._recursive_listdir(self.dir_path, 0)

    def _recursive_listdir(self, path, cur_depth):
        if cur_depth >= self.max_depth:
            raise ValueError(f"recursive list dir error because up to max_depth:{self.max_depth}")
        FileCheck.dir_check(path)

        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)

            if os.path.isfile(file_path):
                self._cur_file_num += 1
                if self._cur_file_num > self.max_file_num:
                    raise ValueError(f"recursive list dir error because up to max file nums:{self.max_file_num}")

                SecFileCheck(file_path, self.max_size).check()

            elif os.path.isdir(file_path):
                self._recursive_listdir(file_path, cur_depth + 1)


class FileCheck:
    MAX_PATH_LENGTH = 1024
    DEFAULT_MAX_FILE_NAME_LEN = 255
    BLACKLIST_PATH = [
        "/etc/",
        "/usr/bin/",
        "/usr/lib/",
        "/usr/lib64/",
        "/sys/",
        "/dev/",
        "/sbin",
        "/tmp"
    ]

    @staticmethod
    def check_file_size(file_path: str, max_file_size: int):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise FileCheckError(f"FileSizeLimit: '{file_path}' size over Limit: {max_file_size}")

    @staticmethod
    def check_input_path_valid(path: str, check_real_path: bool = True, check_blacklist: bool = False):
        if not path or not isinstance(path, str):
            raise FileCheckError(f"Input path '{path}' is not valid str")

        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}'... length over limit")

        if ".." in path:
            raise FileCheckError(f"there are illegal characters in path '{path}'")

        if check_real_path and Path(path).resolve() != Path(path).absolute():
            raise FileCheckError(f"Input path '{path}' is not valid")
        path_obj = Path(path)
        if check_blacklist:
            for black_path in FileCheck.BLACKLIST_PATH:
                if path_obj.resolve().is_relative_to(black_path):
                    raise FileCheckError(f"Input path '{path}' is in blacklist")

    @staticmethod
    def check_path_is_exist_and_valid(path: str, check_real_path: bool = True, check_blacklist: bool = False):
        if not isinstance(path, str):
            raise FileCheckError(f"Input path '{path}' is not valid str")

        if len(path) == 0:
            raise FileCheckError(f"Input path is ''")

        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")

        if not os.path.exists(path):
            raise FileCheckError(f"path '{path}' is not exists")

        FileCheck.check_input_path_valid(path, check_real_path, check_blacklist)

    @staticmethod
    def dir_check(file_path: str):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")

        if not file_path.startswith("/"):
            raise FileCheckError(f"dir '{file_path}' must be an absolute path")

        if not os.path.isdir(file_path):
            raise PathNotDirException(f"PathNotDirException: ['{file_path}'] is not a valid dir")

        FileCheck.check_input_path_valid(file_path, True)

    @staticmethod
    def check_files_num_in_directory(directory_path: str, suffix: str, limit: int):
        if len(directory_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{directory_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        count = sum(1 for file in Path(directory_path).glob("*") if not suffix or file.suffix == suffix)
        if count > limit:
            raise FileCheckError(f"The number of '{suffix}' files in '{directory_path}' exceed {limit}")

    @staticmethod
    def check_file_owner(file_path: str):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        current_user_uid = os.getuid()

        def check_owner(path: str, path_type: str):
            """辅助函数，用于检查一个文件或目录的属主。"""
            try:
                stat_info = os.stat(path)
                owner_uid = stat_info.st_uid
                if owner_uid != current_user_uid:
                    raise FileCheckError(f"The owner of the {path_type} '{path}' is different from the current user")
            except FileNotFoundError as fnf_error:
                raise FileCheckError(f"The {path_type} '{path}' does not exist") from fnf_error
            except PermissionError as pe_error:
                raise FileCheckError(f"Permission denied when accessing the {path_type} '{path}'") from pe_error

        # 检查文件的属主
        check_owner(file_path, "file")

        # 获取文件所在的目录
        dir_path = os.path.dirname(os.path.abspath(file_path))
        # 检查目录的属主
        check_owner(dir_path, "directory")

    @staticmethod
    def check_filename_valid(file_path: str, max_length: int = 0):
        max_length = FileCheck.DEFAULT_MAX_FILE_NAME_LEN if max_length <= 0 else max_length
        file_name = os.path.basename(file_path)
        if len(file_name) > max_length:
            raise FileCheckError(f"the file name length of {file_name[:max_length]}... is over limit {max_length}")

    @staticmethod
    def check_mode(file_path: str, mode_limit=0o755):
        try:
            status = os.stat(file_path)
        except FileNotFoundError as fnf_error:
            raise FileCheckError(f"File not found: {file_path}") from fnf_error
        except PermissionError as pe_error:
            raise FileCheckError(f"Permission denied when accessing the file: {file_path}") from pe_error
        except Exception as e:
            raise FileCheckError(f"get [{file_path}] status failed: {e}") from e
        mode_a = status.st_mode
        mode_b = mode_limit
        # 文件权限比较，权限为3位八进制数，从右往前比较，遇到第一位a比b大，则认为mode_a比mode_b大

        mode_a = mode_a & 0o777
        mode_b = mode_b & 0o777
        if mode_a == mode_b:
            return

        for _ in range(3 * 3):
            if (mode_a & 1) > (mode_b & 1):
                raise FileCheckError(f"the file [{file_path}] mode:[{oct(mode_a)}] greater than [{oct(mode_b)}]")

            mode_a >>= 1
            mode_b >>= 1


def check_disk_free_space(path, volume):
    _, _, free = shutil.disk_usage(path)
    return free < volume


def safetensors_check(mode_path):
    path = Path(mode_path)
    safertensors = any(path.glob('*.safetensors'))
    if not safertensors:
        logger.warning('The current model does not contain model files in satensors format.')
