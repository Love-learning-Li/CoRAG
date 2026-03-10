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

import re
from typing import List, Optional, Callable
from loguru import logger
from sqlalchemy import Engine, Index, select, func, bindparam, text, inspect
from sqlalchemy.exc import SQLAlchemyError

from mx_rag.storage.document_store.models import ChunkModel
from mx_rag.storage.document_store.base_storage import MxDocument, Docstore, StorageError
from mx_rag.storage.document_store.helper_storage import _DocStoreHelper
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, TEXT_MAX_LEN, MAX_TOP_K, validate_list_str, STR_MAX_LEN


class OpenGaussDocstore(Docstore):
    @validate_params(
        engine=dict(validator=lambda x: isinstance(x, Engine), message="Param must be a Engine"),
        encrypt_fn=dict(validator=lambda x: x is None or callable(x), message="Param must be a callable or None"),
        decrypt_fn=dict(validator=lambda x: x is None or callable(x), message="Param must be a callable or None"),
        enable_bm25=dict(validator=lambda x: isinstance(x, bool), message="Param must be a bool"),
        index_name=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                        message="Param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"))
    def __init__(self, engine: Engine, encrypt_fn: Callable = None, decrypt_fn: Callable = None, enable_bm25=True,
                 index_name: str = "chunks_content_bm25"):
        super().__init__()
        if engine.name != "opengauss":
            raise StorageError("engine only support OpenGauss dialect.")
        self.doc_store = _DocStoreHelper(engine, encrypt_fn, decrypt_fn)
        self.engine = engine
        self.index_name = "idx_" + index_name
        self.index = None
        self._enable_bm25 = enable_bm25
        if enable_bm25:
            self.index = Index(index_name, ChunkModel.chunk_content, opengauss_using="bm25")
            if not any(index['name'] == index_name for index in inspect(engine).get_indexes(ChunkModel.__tablename__)):
                self.index.create(engine)
            else:
                logger.warning(f"OpenGauss Index {index_name} already exists")

    def drop(self):
        ids = set(self.get_all_document_id())
        for idx in ids:
            self.delete(idx)
        if self.index:
            self.index.drop(self.engine)

    @validate_params(
        documents=dict(validator=lambda x: isinstance(x, list) and 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
                         message="param must be List[MxDocument] and length range in (0, 1000 * 1000]"),
        document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0,
                         message="param must greater equal than 0")
    )
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
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

    @validate_params(
        query=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
            message=f"param must be str and length range (0, {TEXT_MAX_LEN}]"),
        top_k=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= MAX_TOP_K,
            message="param must be int and must in range (0, 10000]"),
        filter_dict=dict(validator=lambda x: x is None or isinstance(x, dict),
                         message="param filter_dict must be None or dict"))
    def full_text_search(self, query: str, top_k: int = 3, filter_dict: dict = None) -> List[MxDocument]:
        if not self._enable_bm25:
            logger.error("OpenGaussDocstore full_text_search failed due to enable_bm25 is False")
            return []
        self._validate_filter_dict(filter_dict)
        doc_filter = filter_dict.get("document_id", []) if filter_dict else []
        if not doc_filter:
            data = self._do_bm25_search()
        else:
            data = self._do_bm25_search_with_filter(doc_filter)
        params = {
            'question_query': query,
            'top_k': top_k
        }
        with self.doc_store.get_transaction() as session:
            try:
                session.execute(text("SET enable_indexscan = on"))
                session.execute(text("SET enable_seqscan = off"))
                result = session.execute(data, params).fetchall()
            except SQLAlchemyError as e:
                logger.error(f"Database operation failed!! :{e}")
                return []
            except Exception as e:
                logger.error(f"openGauss full text search failed!! :{e}")
                return []
        final_results = []
        for item in result:
            if len(item) < 4:
                logger.warning(f"full_text_search: parse OpenGauss result failed, length of item less 4({len(item)})")
                continue
            item[1]["score"] = item[3]
            final_results.append(MxDocument(
                page_content=item[0],
                metadata=item[1],
                document_name=item[2]
            ))
        return final_results

    def _do_bm25_search(self):
        data = select(
            ChunkModel.chunk_content,
            ChunkModel.chunk_metadata,
            ChunkModel.document_name,
            (ChunkModel.chunk_content.op("<&>")(func.cast(text(":question_query"),
                                                          ChunkModel.chunk_content.type))).label("score")
        ).order_by(
            text("score DESC")
        ).limit(bindparam("top_k"))
        return data

    def _do_bm25_search_with_filter(self, doc_filter: list):
        data = select(
            ChunkModel.chunk_content,
            ChunkModel.chunk_metadata,
            ChunkModel.document_name,
            (ChunkModel.chunk_content.op("<&>")(func.cast(text(":question_query"),
                                                          ChunkModel.chunk_content.type))).label("score")
        ).filter(
            ChunkModel.document_id.in_(doc_filter)
        ).order_by(
            text("score DESC")
        ).limit(bindparam("top_k"))
        return data
