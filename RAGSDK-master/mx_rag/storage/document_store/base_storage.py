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

from loguru import logger
from pydantic import BaseModel, Field, field_validator, ConfigDict

from mx_rag.utils.common import (validate_sequence, MAX_PAGE_CONTENT, MAX_FILTER_SEARCH_ITEM, MAX_CHUNKS_NUM,
                                 MAX_STDOUT_STR_LEN, STR_MAX_LEN)


class StorageError(Exception):
    pass


class MxDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_content: str = Field(max_length=MAX_PAGE_CONTENT)
    metadata: dict = Field(default_factory=dict)
    document_name: str = Field(max_length=1024)

    @field_validator('metadata')
    def _validate_metadata(cls, metadata):
        if not validate_sequence(metadata, STR_MAX_LEN):
            raise ValueError("check MxDocument metadata failed")

        return metadata


class Docstore(ABC):
    @abstractmethod
    def search(self, chunk_id) -> MxDocument:
        pass

    @abstractmethod
    def delete(self, document_id):
        pass

    @abstractmethod
    def add(self, documents, document_id):
        pass

    @abstractmethod
    def get_all_chunk_id(self) -> List[int]:
        pass

    @abstractmethod
    def get_all_document_id(self) -> List[int]:
        pass

    @abstractmethod
    def search_by_document_id(self, document_id: int):
        pass

    @abstractmethod
    def update(self, chunk_ids: List[int], texts: List[str]):
        pass

    def _validate_filter_dict(self, filter_dict):
        if not filter_dict:
            return
        if len(filter_dict) > MAX_FILTER_SEARCH_ITEM:
            raise ValueError(
                f"filter_dict invalid length({len(filter_dict)}) is greater than {MAX_FILTER_SEARCH_ITEM}")
        invalid_keys = str(filter_dict.keys() - {"document_id"})
        if invalid_keys:
            logger.warning(f"{invalid_keys[:MAX_STDOUT_STR_LEN]} ... is no support")
        doc_filter = filter_dict.get("document_id", [])
        if not isinstance(doc_filter, list) or not all(isinstance(item, int) for item in doc_filter):
            raise ValueError("value of 'document_id' in filter_dict must be List[int]")
        doc_filter = list(set(doc_filter))  # 去重
        if len(doc_filter) > MAX_CHUNKS_NUM:
            raise ValueError(f"length of 'document_id' in filter_dict is too large ( > {MAX_CHUNKS_NUM})")
