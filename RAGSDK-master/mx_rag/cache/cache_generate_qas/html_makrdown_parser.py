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

import concurrent
import glob
import os
import re
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from loguru import logger
from tqdm import tqdm

from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP_1024
from mx_rag.utils.file_check import FileCheck, SecFileCheck

MAX_FILE_SIZE_10M = 10 * 1024 * 1024
MAX_FILE_NUM = 1000


class GenerateQaParser(ABC):

    @abstractmethod
    def parse(self):
        pass


def _thread_pool_callback(worker):
    worker_exception = worker.exception()
    if worker_exception:
        logger.error(
            "called thread pool executor callback function, worker return exception: {}".format(worker_exception))


def _md_load(file_path: str) -> List[str]:
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    docs = []
    for document in documents:
        # 过滤markdown中以base64编码的图片内容
        lines = []
        for line in document.page_content.splitlines(keepends=True):
            if "data:image" in line.lower():
                continue
            lines.append(line)
        docs.append(''.join(lines))
    return docs


class MarkDownParser(GenerateQaParser):
    """
    功能描述:
        这是一个用于解析markdown的类，它继承自GenerateQaParser类。
        返回内容为文件名和内容
    Attributes:
        file_path: 需要解析的markdown所在文件夹
        max_file_num: 解析的最大文件数
    """

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024, message=STR_TYPE_CHECK_TIP_1024),
        max_file_num=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= 10000,
                          message="param must be int and value range [1, 10000]")
    )
    def __init__(self, file_path: str, max_file_num: int = MAX_FILE_NUM):
        self.file_path = file_path
        self.max_file_num = max_file_num

    def parse(self) -> Tuple[List[str], List[str]]:
        def _load_file(_mk, progress_bar):
            SecFileCheck(_mk.as_posix(), MAX_FILE_SIZE_10M).check()
            docs = _md_load(_mk.as_posix())
            if not docs:
                return _mk.name, ""
            progress_bar.update(1)
            return _mk.name, docs[0]

        FileCheck.dir_check(self.file_path)
        FileCheck.check_files_num_in_directory(self.file_path, ".md", self.max_file_num)
        titles = []
        contents = []
        task_list = []
        progress_bar = tqdm(total=len(glob.glob(os.path.join(self.file_path, "*.md"))))
        with ThreadPoolExecutor() as executor:
            for _mk in Path(self.file_path).glob("*.md"):
                thread_pool_exc = executor.submit(
                    _load_file,
                    _mk,
                    progress_bar
                )
                thread_pool_exc.add_done_callback(_thread_pool_callback)
                task_list.append(thread_pool_exc)
            failed_count = 0
            for future in concurrent.futures.as_completed(task_list):
                title, content = future.result()
                if not content:
                    failed_count += 1
                    logger.warning(f"skip {failed_count} file, failed to get content.")
                    continue
                titles.append(title)
                contents.append(content)
        return titles, contents
