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

import datetime
import os
import pathlib
import re
from typing import List, Callable, Optional, NoReturn

import numpy as np
import sqlalchemy
from loguru import logger
from sqlalchemy import create_engine, Column, Integer, String, DateTime, UniqueConstraint, func, Enum
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

from mx_rag.knowledge.base_knowledge import KnowledgeBase, KnowledgeError
from mx_rag.storage.document_store import Docstore, MxDocument
from mx_rag.storage.vectorstore import VectorStore
from mx_rag.utils.common import validate_params, FILE_COUNT_MAX, MAX_SQLITE_FILE_NAME_LEN, \
    check_db_file_limit, validate_list_str, TEXT_MAX_LEN, STR_TYPE_CHECK_TIP_1024, validate_sequence, STR_MAX_LEN, \
    check_pathlib_path, validate_lock, BOOL_TYPE_CHECK_TIP, check_embed_func
from mx_rag.utils.file_check import FileCheck, check_disk_free_space

Base = declarative_base()


class KnowledgeModel(Base):
    __tablename__ = "knowledge_table"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(Integer, nullable=False)
    knowledge_name = Column(String, comment="知识库名称")
    user_id = Column(String, comment="用户id")
    role = Column(Enum("admin", "member"), comment="用户角色，admin: 管理员, member: 仅查询")
    create_time = Column(DateTime, comment="创建时间", default=datetime.datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('knowledge_name', 'user_id', name="knowledge_name"),
        {"sqlite_autoincrement": True}
    )


class DocumentModel(Base):
    __tablename__ = "document_table"

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_id = Column(Integer, comment="知识库ID", nullable=False)
    knowledge_name = Column(String, comment="知识库名称")
    document_name = Column(String, comment="文档名称")
    document_file_path = Column(String, comment="文档路径")
    create_time = Column(DateTime, comment="创建时间", default=datetime.datetime.utcnow)
    __table_args__ = (
        UniqueConstraint('knowledge_id', 'document_name', name="knowledge_id"),
        {"sqlite_autoincrement": True}
    )


class KnowledgeStore:
    FREE_SPACE_LIMIT = 200 * 1024 * 1024

    @validate_params(
        db_path=dict(validator=lambda x: 0 < len(x) <= 1024 and isinstance(x, str), message=STR_TYPE_CHECK_TIP_1024)
    )
    def __init__(self, db_path: str):
        FileCheck.check_input_path_valid(db_path, check_blacklist=True)
        FileCheck.check_filename_valid(db_path, MAX_SQLITE_FILE_NAME_LEN)
        self.db_path = db_path
        engine = create_engine(f"sqlite:///{db_path}")
        self.session = scoped_session(sessionmaker(bind=engine))
        Base.metadata.create_all(engine)
        os.chmod(db_path, 0o600)

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                       message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")

    )
    def add_doc_info(self, knowledge_name: str, doc_name: str, file_path: str, user_id: str):
        if check_disk_free_space(os.path.dirname(self.db_path), self.FREE_SPACE_LIMIT):
            logger.error("Insufficient remaining space. Please clear disk space.")
            raise KnowledgeError("Insufficient remaining space, please clear disk space")
        check_db_file_limit(self.db_path)
        with self.session() as session:
            try:
                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
                if knowledge is None:
                    raise KnowledgeError(f"knowledge_name={knowledge_name}, user_id={user_id} does not exist in "
                                         f"knowledge_table, please use add_knowledge or add_usr_id_to_knowledge"
                                         f" function to add them")
                if not self._check_usr_role_is_admin(knowledge_name, user_id):
                    raise KnowledgeError(f"(user_id={user_id}) is not admin, can not add document")
                knowledge_id = knowledge.knowledge_id
                # 创建新的文档
                document_model = DocumentModel(knowledge_id=knowledge_id, knowledge_name=knowledge_name,
                                               document_name=doc_name, document_file_path=file_path)
                session.add(document_model)
                session.commit()
                logger.debug(f"success add (knowledge_name={knowledge_name}, "
                             f"doc_name={doc_name}, user_id={user_id}) in knowledge_table.")
                return document_model.document_id
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(
                    f"Database error while adding knowledge: '{knowledge_name}', document: '{doc_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to add knowledge: '{knowledge_name}', document: '{doc_name}' "
                    "due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"add chunk failed, {err}") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")
    )
    def delete_doc_info(self, knowledge_name: str, doc_name: str, user_id: str):
        if not self._check_usr_role_is_admin(knowledge_name, user_id):
            raise KnowledgeError(f"(user_id={user_id}) is not admin, can not delete document")
        with self.session() as session:
            try:
                knowledge = session.query(KnowledgeModel
                                          ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
                if not knowledge:
                    logger.debug(f"{knowledge_name} does not exist in knowledge_table, no need delete.")
                    return None
                # 删除文档信息
                doc_to_delete = session.query(DocumentModel).filter_by(document_name=doc_name,
                                                                       knowledge_id=knowledge.knowledge_id).first()
                if not doc_to_delete:
                    logger.debug(f"{doc_name} does not exist in {knowledge_name}, no need delete.")
                    return None
                else:
                    session.delete(doc_to_delete)
                    session.commit()
                    logger.debug(f"success delete (knowledge_name={knowledge_name}, "
                                 f"document_name={doc_name}, user_id={user_id}) in document_table.")
                return doc_to_delete.document_id
            except SQLAlchemyError as db_err:
                session.rollback()
                logger.error(
                    f"Database error while deleting knowledge: '{knowledge_name}', document: '{doc_name}': {db_err}")
                raise KnowledgeError(
                    f"Failed to delete knowledge: '{knowledge_name}', document: '{doc_name}' "
                    "due to a database error: {db_err}") from db_err
            except Exception as err:
                session.rollback()
                raise KnowledgeError(f"delete chunk failed, {err}") from err

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")
    )
    def get_all_documents(self, knowledge_name: str, user_id: str):
        with self.session() as session:
            knowledge = session.query(KnowledgeModel
                                      ).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if knowledge is None:
                logger.debug(f"(knowledge_name={knowledge_name}, user_id={user_id}) does not exist in knowledge_table")

            if knowledge:
                return session.query(DocumentModel).filter_by(knowledge_id=knowledge.knowledge_id).all()
            return []

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")
    )
    def check_document_exist(self, knowledge_name: str, doc_name: str, user_id: str) -> bool:
        with self.session() as session:
            # 同一个user_id下知识库名称不能重复
            knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if not knowledge:
                return False
            chunk = session.query(DocumentModel).filter_by(
                knowledge_id=knowledge.knowledge_id, document_name=doc_name).first()
            return chunk is not None

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def _check_usr_role_is_admin(self, knowledge_name: str, user_id: str) -> bool:
        with self.session() as session:
            knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name, user_id=user_id).first()
            if not knowledge:
                raise KnowledgeError(f"(user_id={user_id}) does not exist in knowledge_table {knowledge_name}")
            return knowledge.role == 'admin'

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")
    )
    def get_all_knowledge_info(self, user_id):
        with self.session() as session:
            knowledge_list = session.query(KnowledgeModel).filter_by(user_id=user_id).all()
        return knowledge_list or []

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        role=dict(validator=lambda x: isinstance(x, str) and x in ['admin'],
                  message="param must be  meets: Type is str, and only admin can add knowledge")
    )
    def add_knowledge(self, knowledge_name, user_id, role='admin'):
        with self.session() as session:
            knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name).first()
            if knowledge:
                logger.debug(f"knowledge={knowledge_name} already in knowledge_table, now add user_id={user_id} "
                             f"to knowledge={knowledge_name}")
                self.add_usr_id_to_knowledge(knowledge_name, user_id, role)
                return knowledge.knowledge_id
            max_id = session.query(KnowledgeModel).with_entities(
                func.max(KnowledgeModel.knowledge_id)).scalar() or 0
            knowledge_id = max_id + 1
            knowledge_model = KnowledgeModel(knowledge_id=knowledge_id, knowledge_name=knowledge_name,
                                             user_id=user_id, role=role)
            session.add(knowledge_model)
            session.commit()
            return knowledge_id

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        role=dict(validator=lambda x: isinstance(x, str) and x in ['admin', 'member'],
                  message="param must be  meets: Type is str, one of ['admin', 'member']")
    )
    def add_usr_id_to_knowledge(self, knowledge_name, user_id, role):
        try:
            with self.session() as session:
                knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name)
                if not knowledge:
                    raise KnowledgeError(f"knowledge_name={knowledge_name} does not exist in knowledge_table")
                user_id_in_knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name,
                                                                               user_id=user_id).first()
                if user_id_in_knowledge:
                    logger.debug(f"user_id={user_id} already in knowledge_table")
                    return
                knowledge_model = KnowledgeModel(knowledge_id=knowledge.first().knowledge_id,
                                                 knowledge_name=knowledge_name, user_id=user_id, role=role)
                session.add(knowledge_model)
                session.commit()
        except SQLAlchemyError as db_err:
            logger.error(f"Database error while add {user_id} to {knowledge_name}: {db_err}")
            raise KnowledgeError(
                f"Failed to add {user_id} to {knowledge_name} due to a database error: {db_err}") from db_err
        except Exception as e:
            raise KnowledgeError(f"failed to add {user_id} to {knowledge_name}") from e

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        role=dict(validator=lambda x: isinstance(x, str) and x in ['admin', 'member'],
                  message="param must be  meets: Type is str, one of ['admin', 'member']"),
        force=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def delete_usr_id_from_knowledge(self, knowledge_name, user_id, role, force=False):
        with self.session() as session:
            knowledge = session.query(KnowledgeModel
                                      ).filter_by(knowledge_name=knowledge_name, user_id=user_id, role=role).first()
            if not knowledge:
                raise KnowledgeError(f"(user_id={user_id}, role={role}, knowledge_name={knowledge_name})"
                                     f" does not exist in knowledge_table")
            knowledges = session.query(KnowledgeModel
                                       ).filter_by(knowledge_id=knowledge.knowledge_id).all()
            if len(knowledges) == 1 and not force:
                raise KnowledgeError(
                    f"The knowledge {knowledge_name} now only belongs to user {user_id}, not support delete. "
                    f"please use KnowledgeDB.delete_all to clear, that operation will delete all documents "
                    f"of {knowledge_name}, and the vector database.")

            session.delete(knowledge)
            session.commit()
            logger.debug(f"success delete (knowledge_name={knowledge_name}, user_id={user_id}), "
                         f"role={role} in knowledge_table.")

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def check_knowledge_exist(self, knowledge_name: str, user_id: str) -> bool:
        return knowledge_name in self._get_all_knowledge_name(user_id)

    @validate_params(
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'")
    )
    def _get_all_knowledge_name(self, user_id: str) -> List[str]:
        knowledge_list = self.get_all_knowledge_info(user_id)
        knowledge_name_list = [knowledge.knowledge_name for knowledge in knowledge_list]
        return knowledge_name_list

    @validate_params(
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024)
    )
    def get_all_usr_role_by_knowledge(self, knowledge_name: str) -> dict:
        with self.session() as session:
            knowledge = session.query(KnowledgeModel).filter_by(knowledge_name=knowledge_name)
            if not knowledge:
                return {}
            return {knowledge.user_id: knowledge.role for knowledge in knowledge.all()}


def _check_metadatas(metadatas) -> bool:
    if metadatas is None:
        return True
    if not isinstance(metadatas, list) or not (0 < len(metadatas) <= TEXT_MAX_LEN):
        logger.error(f"metadatas type incorrect or length over {TEXT_MAX_LEN}")
        return False
    for item in metadatas:
        if not isinstance(item, dict):
            logger.error("metadata type is not dict")
            return False
        if not validate_sequence(item, max_str_length=STR_MAX_LEN):
            return False

    return True


def _check_embedding(embed_type, embeddings, texts, metadatas):
    if embed_type == "dense":
        if not (isinstance(embeddings, (List, np.ndarray)) and len(embeddings) > 0 and
                isinstance(embeddings[0], (List, np.ndarray)) and len(embeddings[0]) > 0
                and isinstance(embeddings[0][0], (float, np.floating))):
            raise KnowledgeError("The data type of dense embedding should be List[List[float]]")
    if embed_type == "sparse":
        if not (isinstance(embeddings, List) and len(embeddings) > 0 and
                isinstance(embeddings[0], dict)):
            raise KnowledgeError("The data type of sparse embedding should be List[List[dict]]")
    if not len(texts) == len(metadatas) == len(embeddings):
        raise KnowledgeError(f"texts, metadatas, {embed_type} embeddings expected to be equal length")


class KnowledgeDB(KnowledgeBase):
    @validate_params(
        knowledge_store=dict(validator=lambda x: isinstance(x, KnowledgeStore),
                             message="param must be instance of KnowledgeStore"),
        chunk_store=dict(validator=lambda x: isinstance(x, Docstore),
                         message="param must be instance of Docstore"),
        vector_store=dict(validator=lambda x: isinstance(x, VectorStore),
                          message="param must be instance of VectorStore"),
        knowledge_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                            message=STR_TYPE_CHECK_TIP_1024),
        max_file_count=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= FILE_COUNT_MAX,
                            message=f"param value range must be [1, {FILE_COUNT_MAX}]"),
        user_id=dict(validator=lambda x: isinstance(x, str) and bool(re.fullmatch(r'^[a-zA-Z0-9_-]{6,64}$', x)),
                     message="param must meets: Type is str, match '^[a-zA-Z0-9_-]{6,64}$'"),
        lock=dict(validator=lambda x: x is None or validate_lock(x),
                  message="param must be one of None, multiprocessing.Lock(), threading.Lock()")
    )
    def __init__(
            self,
            knowledge_store: KnowledgeStore,
            chunk_store: Docstore,
            vector_store: VectorStore,
            knowledge_name: str,
            white_paths: List[str],
            user_id: str,
            max_file_count: int = 1000,
            lock=None
    ):
        super().__init__(white_paths)
        self._knowledge_store = knowledge_store
        self._vector_store = vector_store
        self._document_store = chunk_store
        self.max_file_count = max_file_count
        self.knowledge_name = knowledge_name
        self.user_id = user_id
        self.lock = lock
        if self.lock:
            with self.lock:
                self._check_store_accordance()
        else:
            self._check_store_accordance()

    def get_all_documents(self):
        """获取当前已上传的所有文档"""
        return self._knowledge_store.get_all_documents(self.knowledge_name, self.user_id)

    @validate_params(
        file=dict(validator=lambda x: check_pathlib_path(x), message="param check failed, please see the log"),
        texts=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]"),
        embed_func=dict(validator=lambda x: isinstance(x, dict) and check_embed_func(x),
                        message="embed_func must be {'dense': x, 'sparse': y}, "
                                "and xy is callable or None, and cannot be None at the same time."),
        metadatas=dict(validator=lambda x: _check_metadatas(x),
                       message='param must meets: Type is List[dict] or None,'
                               f' list length range [1, {TEXT_MAX_LEN}], other check please see the log')
    )
    def add_file(self, file: pathlib.Path, texts: List[str], embed_func: dict,
                 metadatas: Optional[List[dict]]) -> NoReturn:
        metadatas = metadatas or [{} for _ in texts]
        embeddings = {}
        if embed_func.get("dense"):
            embeddings["dense"] = embed_func["dense"](texts)
            _check_embedding("dense", embeddings["dense"], texts, metadatas)
        if embed_func.get("sparse"):
            embeddings["sparse"] = embed_func["sparse"](texts)
            _check_embedding("sparse", embeddings["sparse"], texts, metadatas)
        documents = [MxDocument(page_content=t, metadata=m, document_name=file.name) for t, m in zip(texts, metadatas)]
        if self.lock:
            with self.lock:
                self._storage_and_vector_add(file.name, file.as_posix(), documents, embeddings)
        else:
            self._storage_and_vector_add(file.name, file.as_posix(), documents, embeddings)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def delete_file(self, doc_name: str):
        if self.lock:
            with self.lock:
                self._storage_and_vector_delete(doc_name)
        else:
            self._storage_and_vector_delete(doc_name)

    def delete_all(self):
        documents = [doc.document_name
                     for doc in self._knowledge_store.get_all_documents(self.knowledge_name, self.user_id)]
        for document in documents:
            self._storage_and_vector_delete(document)
        user_role = self._knowledge_store.get_all_usr_role_by_knowledge(self.knowledge_name)
        for user_id, role in user_role.items():
            self._knowledge_store.delete_usr_id_from_knowledge(self.knowledge_name, user_id, role, True)

    @validate_params(
        doc_name=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                      message=STR_TYPE_CHECK_TIP_1024)
    )
    def check_document_exist(self, doc_name: str) -> bool:
        return self._knowledge_store.check_document_exist(self.knowledge_name, doc_name, self.user_id)

    def _check_store_accordance(self) -> None:
        chunk_ids = set(self._document_store.get_all_chunk_id())
        vec_ids = set(self._vector_store.get_all_ids())
        if chunk_ids != vec_ids:
            logger.warning("vector store does not consistent with the document store,this maybe affect retrieve result")

    def _storage_and_vector_delete(self, doc_name: str):
        document_id = self._knowledge_store.delete_doc_info(self.knowledge_name, doc_name, self.user_id)
        if document_id is None:
            logger.warning(f"doc_name={doc_name} does not exist in knowledge_name={self.knowledge_name}")
            return
        ids = self._document_store.delete(document_id)
        num_removed = self._vector_store.delete(ids)
        if len(ids) != num_removed:
            logger.warning("the number of documents does not match the number of vectors")

    def _storage_and_vector_add(self, doc_name: str, file_path: str, documents: List, embeddings: dict):
        document_id = self._knowledge_store.add_doc_info(self.knowledge_name, doc_name, file_path, self.user_id)
        ids = self._document_store.add(documents, document_id)
        dense_vector = embeddings.get("dense")
        sparse_vector = embeddings.get("sparse")
        if dense_vector and sparse_vector:
            self._vector_store.add_dense_and_sparse(ids, np.array(dense_vector), sparse_vector)
        elif dense_vector:
            self._vector_store.add(ids, np.array(dense_vector), document_id)
        else:
            from mx_rag.storage.vectorstore import MindFAISS
            if isinstance(self._vector_store, MindFAISS):
                raise KnowledgeError("MindFAISS does not support sparse embeddings")
            self._vector_store.add_sparse(ids, sparse_vector)
