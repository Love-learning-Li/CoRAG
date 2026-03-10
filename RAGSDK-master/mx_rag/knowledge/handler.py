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

from pathlib import Path
from typing import Callable, List, Union

from loguru import logger

from mx_rag.document import LoaderMng
from mx_rag.document.loader import BaseLoader
from mx_rag.knowledge.base_knowledge import KnowledgeBase
from mx_rag.knowledge.knowledge import KnowledgeDB
from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, CALLABLE_TYPE_CHECK_TIP, NO_SPLIT_FILE_TYPE, \
    FILE_COUNT_MAX, STR_TYPE_CHECK_TIP_1024, check_embed_func, EMBED_FUNC_TIP
from mx_rag.utils.file_check import SecFileCheck, FileCheck


class FileHandlerError(Exception):
    pass


@validate_params(
    knowledge=dict(validator=lambda x: isinstance(x, KnowledgeDB), message="param must be instance of KnowledgeDB"),
    loader_mng=dict(validator=lambda x: isinstance(x, LoaderMng), message="param must be instance of LoaderMng"),
    embed_func=dict(validator=lambda x: check_embed_func(x), message=EMBED_FUNC_TIP),
    force=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
)
def upload_files(
        knowledge: KnowledgeDB,
        files: List[str],
        loader_mng: LoaderMng,
        embed_func: Union[Callable, dict],
        force: bool = False
):
    """上传单个文档，不支持的文件类型会抛出异常，如果文档重复，可选择强制覆盖"""
    if len(files) > knowledge.max_file_count:
        raise FileHandlerError(f'files list length must less than {knowledge.max_file_count}, upload files failed')

    # 无文件上传时，添加告警提示
    if not files:
        logger.warning("no files need to be loaded")
        return []
    # 传入非字典形式，默认为稠密向量解析
    if not isinstance(embed_func, dict):
        embed_func = {"dense": embed_func, "sparse": None}
    fail_files = []
    for file in files:
        _check_file(file, force, knowledge)
        file_obj = Path(file)

        loader_info = loader_mng.get_loader(file_obj.suffix)

        loader = loader_info.loader_class(file_path=file_obj.as_posix(), **loader_info.loader_params)

        docs = []
        for doc in loader.load():
            if doc.metadata.get("type", "") == "text":
                splitter_info = loader_mng.get_splitter(file_obj.suffix)
                splitter = splitter_info.splitter_class(**splitter_info.splitter_params)
                docs.extend(splitter.split_documents([doc]))
            # 图片不需要切分的直接load
            else:
                docs.append(doc)

        texts = [doc.page_content for doc in docs if doc.page_content]
        meta_data = [doc.metadata for doc in docs if doc.page_content]
        try:
            knowledge.add_file(file_obj, texts, embed_func, meta_data)
        except Exception as err:
            # 当添加文档失败时，删除已添加的部分文档做回滚，捕获异常是为了正常回滚
            fail_files.append(file)
            try:
                knowledge.delete_file(file_obj.name)
            except Exception as e:
                logger.warning(f"exception encountered while rollback, {e}")
            logger.error(f"add '{file}' failed, {err}")
            continue
    if len(fail_files) > 0:
        logger.error(f"These files '{fail_files}' add failed")
    return fail_files


def _check_file(file: str, force: bool, knowledge: KnowledgeBase):
    """
    检查文件路径
    """
    SecFileCheck(file, BaseLoader.MAX_SIZE).check()
    file_obj = Path(file)
    if not _is_in_white_paths(file_obj, knowledge.white_paths):
        raise FileHandlerError(f"'{file_obj.as_posix()}' is not in whitelist path")
    if not file_obj.is_file():
        raise FileHandlerError(f"'{file}' is not a normal file")
    if knowledge.check_document_exist(file_obj.name):
        if not force:
            raise FileHandlerError(f"file path '{file_obj.name}' is already exist")
        else:
            logger.info(f"delete files '{file_obj.name}' as file is already exist")
            knowledge.delete_file(file_obj.name)


def _is_in_white_paths(file_obj: Path, white_paths: List[str]) -> bool:
    """
    判断是否在白名单路径中
    """
    for p in white_paths:
        if file_obj.resolve().is_relative_to(p):
            return True
    return False


class FilesLoadInfo:
    @validate_params(
        knowledge=dict(validator=lambda x: isinstance(x, KnowledgeDB), message="param must be instance of KnowledgeDB"),
        dir_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024, message=STR_TYPE_CHECK_TIP_1024),
        loader_mng=dict(validator=lambda x: isinstance(x, LoaderMng), message="param must be instance of LoaderMng"),
        embed_func=dict(validator=lambda x: isinstance(x, Callable), message=CALLABLE_TYPE_CHECK_TIP),
        force=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        load_image=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self,
                 knowledge: KnowledgeDB,
                 dir_path: str,
                 loader_mng: LoaderMng,
                 embed_func: Callable[[List[str]], List[List[float]]],
                 force: bool = False,
                 load_image: bool = False):
        self.knowledge = knowledge
        self.dir_path = dir_path
        self.loader_mng = loader_mng
        self.embed_func = embed_func
        self.force = force
        self.load_image = load_image


@validate_params(
    params=dict(validator=lambda x: isinstance(x, FilesLoadInfo),
                message="param must be instance of FilesLoadInfo")
)
def upload_dir(params: FilesLoadInfo):
    knowledge = params.knowledge
    dir_path = params.dir_path
    loader_mng = params.loader_mng
    embed_func = params.embed_func
    force = params.force
    load_image = params.load_image
    """
    只遍历当前目录下的文件，不递归查找子目录文件;目录中不支持的文件类型会跳过,但是因注册错误的类型，导致解析过程中错误会中断;如果文档重复，可选择强制覆盖;
    超过最大文件数量则退出;load_image为True时只支持图片类型, False时只支持文档类型;支持图片类型时,无需注册spliter;支持文档类型时必须注册spliter;
    embedding模型需要与支持的文件类型一致,否则会出现embedding错误.
    """
    FileCheck.dir_check(dir_path)
    FileCheck.check_files_num_in_directory(dir_path, "", FILE_COUNT_MAX)
    loader_types = []
    for file_types, _ in loader_mng.loaders.items():
        loader_types.append(file_types)
    spliter_types = []
    for file_types, _ in loader_mng.splitters.items():
        spliter_types.append(file_types)
    if not load_image:
        support_file_type = list(set(loader_types) & set(spliter_types))
    else:
        support_file_type = list(set(loader_types) & set(NO_SPLIT_FILE_TYPE))
    count = 0
    files = []
    unsupported_files = []
    for file in Path(dir_path).glob("*"):
        if count >= knowledge.max_file_count:
            raise FileHandlerError(f'The number of files in the {dir_path} must less than'
                                   f' {knowledge.max_file_count}, upload dir failed')
        if file.suffix in support_file_type:
            files.append(file.as_posix())
            count += 1
        else:
            unsupported_files.append(file.as_posix())
    if len(unsupported_files) > 0:
        logger.warning(f"These files '{unsupported_files}' are not of supported types "
                       f"because no loader or splitter has been registered.")
    fail_files = upload_files(knowledge, files, loader_mng, embed_func, force)

    return unsupported_files + fail_files


@validate_params(
    knowledge=dict(validator=lambda x: isinstance(x, KnowledgeDB), message="param must be instance of KnowledgeDB"),
)
def delete_files(
        knowledge: KnowledgeDB,
        doc_names: List[str]
):
    """删除上传的文档，需传入待删除的文档名称"""
    if len(doc_names) > knowledge.max_file_count:
        raise FileHandlerError(f'files list length must less than {knowledge.max_file_count}, delete files failed')
    if not isinstance(doc_names, list):
        raise FileHandlerError(f"files param {doc_names} is invalid")

    if not doc_names:
        logger.warning("no docs need to be deleted")
        return

    for doc_name in doc_names:
        if not isinstance(doc_name, str):
            raise FileHandlerError(f"file path '{doc_name}' is invalid")
        knowledge.delete_file(doc_name)
