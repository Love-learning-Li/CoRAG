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
from typing import List, Optional, Callable

from sqlalchemy import URL, create_engine

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import StorageError, Docstore
from mx_rag.storage.document_store.helper_storage import _DocStoreHelper
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, MAX_SQLITE_FILE_NAME_LEN, \
    check_db_file_limit, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN
from mx_rag.utils.file_check import FileCheck, check_disk_free_space


class SQLiteDocstore(Docstore):
    FREE_SPACE_LIMIT = 200 * 1024 * 1024
    MAX_DOC_NAME_LEN = 1024

    @validate_params(
        encrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="encrypt_fun must be None or callable function"),
        decrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="decrypt_fun must be None or callable function")
    )
    def __init__(self, db_path: str, encrypt_fn: Callable = None, decrypt_fn: Callable = None):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        FileCheck.check_filename_valid(db_path, max_length=MAX_SQLITE_FILE_NAME_LEN)
        self.db_path = db_path
        engine = create_engine(url=URL.create("sqlite", database=db_path))
        self.doc_store = _DocStoreHelper(engine, encrypt_fn, decrypt_fn)
        os.chmod(db_path, 0o600)

    @validate_params(
        documents=dict(
            validator=lambda x: isinstance(x, list) and 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
            message="param must be List[MxDocument] and length range in (0, 1000 * 1000]"),
        document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0,
                         message="param must greater equal than 0")
    )
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
        FileCheck.check_input_path_valid(self.db_path, check_blacklist=True)
        FileCheck.check_filename_valid(self.db_path, max_length=MAX_SQLITE_FILE_NAME_LEN)
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            raise StorageError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        return self.doc_store.add(documents, document_id)

    @validate_params(document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message="param must greater equal than 0"))
    def delete(self, document_id: int) -> List[int]:
        return self.doc_store.delete(document_id)

    @validate_params(chunk_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message="param must greater equal than 0"))
    def search(self, chunk_id: int) -> Optional[MxDocument]:
        return self.doc_store.search(chunk_id)

    def get_all_chunk_id(self) -> List[int]:
        return self.doc_store.get_all_chunk_id()

    def get_all_document_id(self) -> List[int]:
        return self.doc_store.get_all_document_id()

    @validate_params(document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message=f"document_id must >= 0"))
    def search_by_document_id(self, document_id: int):
        return self.doc_store.search_by_document_id(document_id)

    @validate_params(
        chunk_ids=dict(validator=lambda x: isinstance(x, list) and 0 < len(x) <= MAX_CHUNKS_NUM,
                       message=f"param value range (0, {MAX_CHUNKS_NUM}]"),
        texts=dict(validator=lambda x: validate_list_str(x, [1, MAX_CHUNKS_NUM], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           f"list length range [1, {MAX_CHUNKS_NUM}], str length range [1, {STR_MAX_LEN}]"),
    )
    def update(self, chunk_ids: List[int], texts: List[str]):
        self.doc_store.update(chunk_ids, texts)
