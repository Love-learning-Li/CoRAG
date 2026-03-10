#!/usr/bin/env python3
# encoding: utf-8
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

from typing import Dict, Any, List, Type
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters.base import TextSplitter
from loguru import logger

from mx_rag.utils.common import (validate_list_str, validate_params, NO_SPLIT_FILE_TYPE,
                                 FILE_TYPE_COUNT, validate_sequence)


class LoaderInfo:
    def __init__(self,
                 loader_class: Type,
                 loader_params: Dict[str, Any]):
        self.loader_class = loader_class
        self.loader_params = loader_params


class SplitterInfo:
    def __init__(self,
                 splitter_class: Type,
                 splitter_params: Dict[str, Any]):
        self.splitter_class = splitter_class
        self.splitter_params = splitter_params


class LoaderMng:
    MAX_REGISTER_LOADER_NUM = 1000
    MAX_REGISTER_SPLITTER_NUM = 1000

    def __init__(self):
        self.loaders: Dict[str, LoaderInfo] = {}
        self.splitters: Dict[str, SplitterInfo] = {}

    @validate_params(
        loader_class=dict(validator=lambda x: issubclass(x, BaseLoader),
                          message="param must be a subclass of BaseLoader in "
                                  "langchain_community.document_loaders.base"),
        file_types=dict(validator=lambda x: validate_list_str(x, [1, FILE_TYPE_COUNT], [1, FILE_TYPE_COUNT]),
                        message="param must meets: Type is List[str], "
                                "list length range [1, 32], str length range [1, 32]"),
        loader_params=dict(validator=lambda x: (isinstance(x, Dict) and validate_sequence(x)) or x is None,
                           message="param must meets: Type must be Dict or None,"
                                   " other check please see the log")
    )
    def register_loader(self, loader_class: BaseLoader, file_types: List[str],
                        loader_params: Dict[str, Any] = None):
        if len(self.loaders) >= self.MAX_REGISTER_LOADER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_LOADER_NUM} loaders are registered")
        for file_type_str in file_types:
            if file_type_str in self.loaders:
                logger.warning(f"the loader class for file type '{file_type_str}' has been updated "
                               f"from '{self.loaders[file_type_str].loader_class}' to '{loader_class}'")
            self.loaders[file_type_str] = LoaderInfo(loader_class, loader_params or {})

    @validate_params(
        splitter_class=dict(validator=lambda x: issubclass(x, TextSplitter),
                            message="param must be a subclass of TextSplitter in langchain_text_splitters.base"),
        file_types=dict(validator=lambda x: validate_list_str(x, [1, FILE_TYPE_COUNT], [1, FILE_TYPE_COUNT]),
                        message="param must meets: Type is List[str], "
                                "list length range [1, 32], str length range [1, 32]")
    )
    def register_splitter(self, splitter_class: TextSplitter, file_types: List[str],
                          splitter_params: Dict[str, Any] = None):
        if splitter_params is not None and not (isinstance(splitter_params, Dict)
                                       and validate_sequence(splitter_params, max_check_depth=2)):
            raise ValueError("invalid splitter_params.")
        if len(self.splitters) >= self.MAX_REGISTER_SPLITTER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_SPLITTER_NUM} splitters are registered")
        if bool(set(NO_SPLIT_FILE_TYPE) & set(file_types)):
            raise KeyError(f"Unsupported register splitter for file type {set(NO_SPLIT_FILE_TYPE) & set(file_types)}")
        for file_type_str in file_types:
            if file_type_str in self.splitters:
                logger.warning(f"the splitter class for file type '{file_type_str}' has been updated "
                               f"from '{self.splitters[file_type_str].splitter_class}' to '{splitter_class}'")
            self.splitters[file_type_str] = SplitterInfo(splitter_class, splitter_params or {})

    @validate_params(
        file_suffix=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= FILE_TYPE_COUNT,
                         message="param must be str, length range [1, 32]"))
    def get_loader(self, file_suffix: str) -> LoaderInfo:
        loader_info = self.loaders.get(file_suffix)
        if loader_info is None:
            raise KeyError(f"No loader registered for file type '{file_suffix}'")
        return loader_info

    @validate_params(
        file_suffix=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= FILE_TYPE_COUNT,
                         message="param must be str, length range [1, 32]"))
    def get_splitter(self, file_suffix: str) -> SplitterInfo:
        splitter_info = self.splitters.get(file_suffix)
        if splitter_info is None:
            raise KeyError(f"No splitter registered for file type '{file_suffix}'")
        return splitter_info

    @validate_params(
        loader_class=dict(validator=lambda x: issubclass(x, BaseLoader),
                          message="param must be langchain_community BaseLoader subclass"),
        file_suffix=dict(validator=lambda x: (isinstance(x, str) and 0 < len(x) <= FILE_TYPE_COUNT) or x is None,
                         message="param must be str, length range [1, 32], or None")
    )
    def unregister_loader(self, loader_class: Type, file_suffix: str = None):
        keys_delete = []
        if file_suffix:
            if file_suffix in self.loaders and self.loaders[file_suffix].loader_class == loader_class:
                keys_delete.append(file_suffix)
            else:
                raise KeyError(f"file type '{file_suffix}': loader class '{loader_class}' is not registered")
        else:
            # 如果file_suffix为空，删除 value 为 loader_class 的所有元素
            for file, loader_info in self.loaders.items():
                if loader_info.loader_class == loader_class:
                    keys_delete.append(file)
            if not keys_delete:
                raise KeyError(f"loader class '{loader_class}' is not registered")

        for key in keys_delete:
            del self.loaders[key]

    @validate_params(
        splitter_class=dict(validator=lambda x: issubclass(x, TextSplitter),
                            message="param must be langchain_community TextSplitter subclass"),
        file_suffix=dict(validator=lambda x: (isinstance(x, str) and 0 < len(x) <= FILE_TYPE_COUNT) or x is None,
                         message="param must be str, length range [1, 32], or None")
    )
    def unregister_splitter(self, splitter_class: Type, file_suffix: str = None):
        keys_delete = []
        if file_suffix:
            if file_suffix in self.splitters and self.splitters[file_suffix].splitter_class == splitter_class:
                keys_delete.append(file_suffix)
            else:
                raise KeyError(f"file type '{file_suffix}': splitter class '{splitter_class}' is not registered")
        else:
            # 删除 value 为 splitter_class 的所有元素
            for file, splitter_info in self.splitters.items():
                if splitter_info.splitter_class == splitter_class:
                    keys_delete.append(file)
            if not keys_delete:
                raise KeyError(f"splitter class '{splitter_class}' is not registered")
        for key in keys_delete:
            del self.splitters[key]
