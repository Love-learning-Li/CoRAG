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


import json
import os
import re
import stat

from loguru import logger

from .common import MAX_FILE_SIZE
from .file_check import FileCheck, SecFileCheck

R_FLAGS = os.O_RDONLY
W_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
A_FLAGS = os.O_RDWR | os.O_CREAT
MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH


def write_jsonl_to_file(datas: list[dict], file: str, flag: str = 'w'):
    if os.path.exists(file):
        SecFileCheck(file, MAX_FILE_SIZE).check()
    else:
        file_path = os.path.dirname(file)
        FileCheck.dir_check(file_path)

    file_basename = os.path.basename(file)
    if not file_basename.endswith('.jsonl'):
        raise Exception(f"file '{file}' is not a jsonl file name")

    try:
        _write_jsonl_file(datas, file, flag)
    except IOError as io_error:
        logger.error(f"write data to file failed, find IOError: {io_error}")
        raise Exception(f"write jsonl to file IO Error") from io_error
    except Exception as e:
        logger.error(f"write data to file failed, find Exception: {e}")
        raise Exception(f"write jsonl to file failed") from e
    logger.info(f"write data to file '{file_basename}' success")


def _write_jsonl_file(datas, file, flag):
    flags = A_FLAGS if flag == 'a' else W_FLAGS
    with os.fdopen(os.open(file, flags, MODES), flag) as f:
        for data in datas:
            data_str = json.dumps(data, ensure_ascii=False)
            f.write(data_str)
            f.write("\n")


def read_jsonl_from_file(file: str,
                         file_size: int = 10 * 1024 * 1024 * 1024):

    SecFileCheck(file, file_size).check()

    datas = []
    try:
        datas = _read_jsonl_file(file)
    except json.JSONDecodeError as json_decode_e:
        logger.error(f"read data from file failed, find JSONDecodeError: {json_decode_e}")
    except Exception as e:
        logger.error(f"read data from file failed, find Exception: {e}")
    return datas


def _read_jsonl_file(file):
    datas = []
    with os.fdopen(os.open(file, R_FLAGS, MODES), 'r') as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            datas.append(data)

            line = f.readline()

    return datas