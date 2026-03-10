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

from contextlib import contextmanager
from typing import List, Optional, Callable, Iterator, Iterable
from sqlalchemy import delete, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from loguru import logger

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import StorageError, Docstore
from mx_rag.storage.document_store.models import Base, ChunkModel
from mx_rag.utils.common import MAX_CHUNKS_NUM, STR_MAX_LEN, MAX_PAGE_CONTENT


class _DocStoreHelper(Docstore):
    def __init__(
            self,
            engine: Engine,
            encrypt_fn: Optional[Callable[[str], str]] = None,
            decrypt_fn: Optional[Callable[[str], str]] = None,
            batch_size: int = 500
    ):
        """
        文档存储实现

        Args:
            engine: 数据库引擎
            encrypt_fn: 内容加密函数 (str -> str)
            decrypt_fn: 内容解密函数 (str -> str)
            batch_size: 批量操作大小
        """
        self.engine = engine
        self.session_factory = scoped_session(
            sessionmaker(
                bind=self.engine,
                autoflush=False,
                expire_on_commit=False
            )
        )
        self._init_db()
        self.batch_size = batch_size
        self.encrypt_fn = encrypt_fn
        self.decrypt_fn = decrypt_fn

    def add(
            self,
            documents: List[MxDocument],
            document_id: int
    ) -> List[int]:
        """分批次添加文档块"""
        if not 0 < len(documents) <= MAX_CHUNKS_NUM:
            raise ValueError(f"Documents count must be between 1 and {MAX_CHUNKS_NUM}")

        def batch_insert(chunk_batch, session):
            # 构造模型对象时同步加密
            chunks = []
            for doc in chunk_batch:
                chunk = ChunkModel(
                    document_id=document_id,
                    document_name=doc.document_name,
                    chunk_content=self._encrypt(doc.page_content),
                    chunk_metadata=doc.metadata
                )
                chunks.append(chunk)
            session.bulk_save_objects(chunks, return_defaults=True)

        try:
            # 分批次处理原始文档
            self._batch_operation(
                iterable=documents,
                operation=batch_insert,
                desc=f"for document {document_id}"
            )

            # 获取生成的ID需要特殊处理（批量插入返回ID的限制）
            with self._transaction() as session:
                last_chunk = session.query(ChunkModel) \
                    .filter_by(document_id=document_id) \
                    .order_by(ChunkModel.chunk_id.desc()) \
                    .limit(len(documents)).all()
                inserted_ids = [c.chunk_id for c in reversed(last_chunk)]

            logger.info("Inserted {} chunks for doc {}", len(inserted_ids), document_id)
            return inserted_ids

        except SQLAlchemyError as e:
            raise StorageError(f"Bulk insert failed: {e}") from e

    def delete(self, document_id: int) -> List[int]:
        """分批次删除文档"""
        try:
            # 先查询所有需要删除的ID
            with self._transaction() as session:
                target_ids = session.query(ChunkModel.chunk_id) \
                    .filter_by(document_id=document_id) \
                    .all()
                target_ids = [id_[0] for id_ in target_ids]

            # 分批次执行删除
            def batch_delete(id_batch, session):
                session.execute(
                    delete(ChunkModel)
                    .where(ChunkModel.chunk_id.in_(id_batch))
                )

            self._batch_operation(
                iterable=target_ids,
                operation=batch_delete,
                desc=f"deleting doc {document_id}"
            )

            logger.info("Deleted {} chunks for doc {}", len(target_ids), document_id)
            return target_ids

        except SQLAlchemyError as e:
            raise StorageError(f"Delete failed: {e}") from e

    def search(self, chunk_id: int) -> Optional[MxDocument]:
        """
        根据chunk_id检索文档

        Args:
            chunk_id: 要查询的块ID

        Returns:
            MxDocument对象或None
        """
        with self._transaction() as session:
            chunk = session.get(ChunkModel, chunk_id)
            if not chunk:
                return None
            return MxDocument(
                page_content=self._decrypt(chunk.chunk_content),
                metadata=chunk.chunk_metadata,
                document_name=chunk.document_name
            )

    def get_all_chunk_id(self) -> List[int]:
        """获取所有chunk_id的生成器实现"""
        with self._transaction() as session:
            query = session.query(ChunkModel.chunk_id).yield_per(self.batch_size)
            return [chunk_id for (chunk_id,) in query]

    def get_all_document_id(self) -> List[int]:
        """获取所有document_id的生成器实现"""
        with self._transaction() as session:
            query = session.query(ChunkModel.document_id).yield_per(self.batch_size)
            return list(set([document_id for (document_id,) in query]))

    def search_by_document_id(self, document_id: int) -> Optional[List[MxDocument]]:
        """通过document_id来获取"""
        with self._transaction() as session:
            chunks = session.query(ChunkModel).filter_by(document_id=document_id).all()
            documents = [MxDocument(
                page_content=self._decrypt(chunk.chunk_content),
                metadata=chunk.chunk_metadata,
                document_name=chunk.document_name
            ) for chunk in chunks]
            return documents

    def update(self, chunk_ids: List[int], texts: List[str]):
        if len(chunk_ids) != len(texts):
            raise StorageError("chunk_ids and texts length not the same while calling update function.")
        with self._transaction() as session:
            updates = [{"chunk_id": chunk_id, "chunk_content": self._encrypt(text)}
                       for chunk_id, text in zip(chunk_ids, texts)]
            session.bulk_update_mappings(ChunkModel, updates)
        logger.info(f"Successfully updated chunk ids {chunk_ids}")

    @contextmanager
    def _transaction(self) -> Iterator[scoped_session]:
        """提供事务上下文管理的会话"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Database operation failed: {}", e)
            raise StorageError(f"Database operation failed: {e}") from e
        except Exception as e:
            session.rollback()
            logger.error("An unexpected error occurred: {}", e)
            raise StorageError(f"Unexpected error occurred: {e}") from e
        finally:
            session.close()

    def get_transaction(self):
        return self._transaction()

    def _init_db(self):
        """初始化数据库表结构"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables initialized")
        except SQLAlchemyError as e:
            logger.critical("Database initialization failed: {}", e)
            raise StorageError("Database setup failed") from e

    def _batch_operation(self, iterable: Iterable, operation: Callable, desc: str = ""):
        """通用分批次操作执行器"""

        total = 0
        batch = []

        def commit_batch(session: Session):
            nonlocal batch, total
            if batch:
                operation(batch, session)
                session.commit()
                total += len(batch)
                logger.debug(f"Processed {total} items {desc}")
                batch = []

        def commit_all(iterable: Iterable, session: Session):
            nonlocal batch
            for i, item in enumerate(iterable, 1):
                batch.append(item)
                if i % self.batch_size == 0:
                    commit_batch(session)
            commit_batch(session)  # 提交最后一批

        try:
            with self._transaction() as session:  # 使用统一的会话上下文
                commit_all(iterable, session)
                logger.info(f"Successfully processed {total} items {desc}")
                return total
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed at {total}: {str(e)}")
            raise StorageError(f"Database operation failed: {e}") from e
        except Exception as e:
            logger.error(f"Batch operation failed at {total}: {str(e)}")
            raise StorageError(f"Batch operation failed: {e}") from e

    def _encrypt(self, text):
        if self.encrypt_fn is not None:
            result = self.encrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= STR_MAX_LEN:
                return result
            else:
                raise ValueError(f"callback function {self.encrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {STR_MAX_LEN}.")
        else:
            return text

    def _decrypt(self, text):
        if self.decrypt_fn is not None:
            result = self.decrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= MAX_PAGE_CONTENT:
                return result
            else:
                raise ValueError(f"callback function {self.decrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {MAX_PAGE_CONTENT}.")

        else:
            return text
