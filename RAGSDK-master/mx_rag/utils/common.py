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

import _thread
import functools
import inspect
import json
import multiprocessing.synchronize
import os
import pathlib
import stat
from datetime import datetime
from enum import Enum
from json import JSONDecodeError
from typing import List, Union, Callable, Dict, Optional, Tuple, Any

import numpy as np
from OpenSSL import crypto
from langchain_core.documents import Document
from loguru import logger

from mx_rag.utils.file_check import FileCheck

FILE_COUNT_MAX = 8000
INT_32_MAX = 2 ** 31 - 1
MAX_DEVICE_ID = 63
MAX_TOP_K = 10000
MAX_QUERY_LENGTH = 128 * 1024 * 1024
EMBEDDING_TEXT_COUNT = 1000 * 1000
EMBEDDING_IMG_COUNT = 1000
IMG_EMBEDDING_TEXT_LEN = 256
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_SPLIT_SIZE = 100 * 1024 * 1024
TEXT_MAX_LEN = 1000 * 1000
STR_MAX_LEN = 128 * 1024 * 1024
MAX_VEC_DIM = 1024 * 1024
NODE_MAX_TEXT_LENGTH = 128 * 1024 * 1024
MILVUS_INDEX_TYPES = ["FLAT"]
MILVUS_METRIC_TYPES = ["L2", "IP", "COSINE"]
MAX_API_KEY_LEN = 128
MAX_PATH_LENGTH = 1024
FILE_TYPE_COUNT = 32
MAX_SQLITE_FILE_NAME_LEN = 200
MIN_SQLITE_FREE_SPACE = 200 * 1024 * 1024

MAX_PROMPT_LENGTH = 1 * 1024 * 1024
MAX_DOCS_COUNT = 1000
MAX_URL_LENGTH = 128
MAX_MODEL_NAME_LENGTH = 128
MAX_TOKEN_NUM = 100000
MAX_BATCH_SIZE = 1024
MAX_ROW_NUM = 10000
MAX_COL_NUM = 1000
MAX_NODE_MUM = 10000

MAX_IMAGE_PIXELS = 4096 * 4096
MIN_IMAGE_PIXELS = 64 * 64
MIN_IMAGE_WIDTH = 256
MIN_IMAGE_HEIGHT = 256
MAX_BASE64_SIZE = 2 * 1024 * 1024
KB = 1024
MB = 1048576  # 1024 * 1024
GB = 1073741824  # 1024 * 1024 * 1024
STR_TYPE_CHECK_TIP = "param must be str"
BOOL_TYPE_CHECK_TIP = "param must be bool"
DICT_TYPE_CHECK_TIP = "param must be dict"
INT_RANGE_CHECK_TIP = "param must be int and value range (0, 2**31-1]"
CALLABLE_TYPE_CHECK_TIP = "param must be callable function"
STR_LENGTH_CHECK_1024 = "param length range [1, 1024]"
STR_TYPE_CHECK_TIP_1024 = "param must be str, length range [1, 1024]"
EMBED_FUNC_TIP = ("embed_func must be callable or {'dense': x, 'sparse': y}, "
                  "and xy is callable or None, and cannot be None at the same time.")
NO_SPLIT_FILE_TYPE = [".jpg", ".png"]
IMAGE_TYPE = (".jpg", ".png")
HEADER_MARK = "#"
DB_FILE_LIMIT = 100 * 1024 * 1024 * 1024
GRAPH_FILE_LIMIT = 10 * 1024 * 1024 * 1024
MAX_CHUNKS_NUM = 1000 * 1000
MAX_PAGE_CONTENT = 16 * MB

MAX_COLLECTION_NAME_LENGTH = 1024

MAX_FILTER_SEARCH_ITEM = 32
MAX_STDOUT_STR_LEN = 1024
MAX_EMBEDDINGS_SIZE = 1024 * 1024
MAX_IDS_SIZE = 1000 * 10000

MAX_RECURSION_LIMIT = 256


def safe_get(data, keys, default=None):
    """
    安全地获取嵌套字典或列表中的值。

    :param data: 字典或列表数据
    :param keys: 键或索引列表，表示嵌套层级
    :param default: 如果键或索引不存在，返回的默认值
    :return: 对应键或索引的值或默认值
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
            data = data[key]
        else:
            return default
    return data


def _get_value_from_param(arg_name, func, *args, **kwargs):
    sig = inspect.signature(func)
    # 从传入参数中获取要校验的value
    for param_name, param in sig.bind(*args, **kwargs).arguments.items():
        if arg_name == param_name:
            return param
    # 传入参数中没有则从方法定义中取默认值
    for name, param in sig.parameters.items():
        if arg_name == name:
            return param.default
    # 都没有抛出异常
    raise ValueError(f"Required parameter '{arg_name}' of function {func.__name__} is missing.")


def validate_params(**validators):
    """
    定义一个装饰器，用于验证函数的多个参数。在方法上使用注释
    @validate_params(
        name=dict(validator=lambda x: isinstance(x, str)),
        age=dict(validator=lambda x: 10 <= x <= 30)
    )
    :param validators: 一个包含验证函数的字典，每个函数用于验证一个特定的参数。
    :return: 装饰器函数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 对每个参数应用验证函数
            for arg_name, validator in validators.items():
                # 检查是否通过位置或关键字传递了参数
                value = _get_value_from_param(arg_name, func, *args, **kwargs)
                # 运行验证函数
                try:
                    if not validator['validator'](value):
                        raise ValueError(f"The parameter '{arg_name}' of function '{func.__name__}' "
                                         f"is invalid, message: {validator.get('message')}")
                except Exception as e:
                    raise ValueError(f"An exception occur during check parameter. "
                                     f"The parameter '{arg_name}' of function '{func.__name__}' "
                                     f"is invalid, message: {validator.get('message')}") from e
            # 如果所有参数都通过验证，则调用原始函数
            return func(*args, **kwargs)

        return wrapper

    return decorator


class PubkeyType(Enum):
    EVP_PKEY_RSA = 6
    EVP_PKEY_DSA = 116
    EVP_PKEY_DH = 28
    EVP_PKEY_EC = 408


class Lang(Enum):
    EN: str = 'en'
    CH: str = 'ch'


class ParseCertInfo:
    """解析根证书信息类"""

    def __init__(self, cert_buffer: str):
        if not cert_buffer:
            raise ValueError("Cert buffer is null.")

        self.cert_info = crypto.load_certificate(crypto.FILETYPE_PEM, str.encode(cert_buffer))
        self.serial_num = hex(self.cert_info.get_serial_number())[2:].upper()
        self.subject_components = self.cert_info.get_subject().get_components()
        self.issuer_components = self.cert_info.get_issuer().get_components()
        self.fingerprint = self.cert_info.digest("sha256").decode()
        self.start_time = datetime.strptime(self.cert_info.get_notBefore().decode(), '%Y%m%d%H%M%SZ')
        self.end_time = datetime.strptime(self.cert_info.get_notAfter().decode(), '%Y%m%d%H%M%SZ')
        self.signature_algorithm = self.cert_info.get_signature_algorithm().decode()
        self.signature_len = self.cert_info.get_pubkey().bits()
        self.cert_version = self.cert_info.get_version() + 1
        self.pubkey_type = self.cert_info.get_pubkey().type()
        self.ca_pub_key = self.cert_info.get_pubkey().to_cryptography_key()
        self.extensions = {}
        for i in range(self.cert_info.get_extension_count()):
            ext = self.cert_info.get_extension(i)
            ext_name = ext.get_short_name().decode()
            try:
                self.extensions[ext_name] = str(ext)
            except (TypeError, ValueError) as e:
                logger.warning(f"Type error or value error, format {ext_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"format '{ext_name}' str info in certificate failed: {e}")
                continue

    @property
    def subject(self) -> str:
        return ", ".join([f"{item[0].decode()}={item[1].decode()}" for item in self.subject_components])

    @property
    def issuer(self) -> str:
        return ", ".join([f"{item[0].decode()}={item[1].decode()}" for item in self.issuer_components])

    def to_dict(self) -> dict:
        return {
            "SerialNum": self.serial_num,
            "Subject": self.subject,
            "Issuer": self.issuer,
            "Fingerprint": self.fingerprint,
            "Date": f"{self.start_time}--{self.end_time}",
        }


def validate_list_document(texts, length_limit: List[int], content_limit: List[int]):
    """
    用于List[Document]类型的数据校验
    Args:
        texts: 输入检索返回Document列表
        length_limit: 列表长度范围
        content_limit: page_content长度范围

    Returns:

    """
    if not isinstance(texts, List):
        logger.error("input is not type List[Document]")
        return False
    min_length_limit = length_limit[0]
    max_length_limit = length_limit[1]
    min_content_limit = content_limit[0]
    max_content_limit = content_limit[1]
    if not min_length_limit <= len(texts) <= max_length_limit:
        logger.error(f"The List[Document] length not in [{min_length_limit}, {max_length_limit}]")
        return False
    for text in texts:
        if not isinstance(text, Document):
            logger.error("The element in the list is not a Document.")
            return False
        if not min_content_limit <= len(text.page_content) <= max_content_limit:
            logger.error(f"The element in List[Document] length not in [{min_content_limit}, {max_content_limit}]")
            return False
    return True


def validate_list_str(texts, length_limit: List[int], str_limit: List[int]):
    """
    用于List[str]类型的数据校验
    Args:
        texts: 输入数据字符串列表
        length_limit: 列表长度范围
        str_limit: 字符串长度范围

    Returns:

    """
    if not isinstance(texts, List):
        logger.error("input is not type List[str]")
        return False
    min_length_limit = length_limit[0]
    max_length_limit = length_limit[1]
    min_str_limit = str_limit[0]
    max_str_limit = str_limit[1]
    if not min_length_limit <= len(texts) <= max_length_limit:
        logger.error(f"The List[str] length not in [{min_length_limit}, {max_length_limit}]")
        return False
    for text in texts:
        if not isinstance(text, str):
            logger.error("The element in the list is not a string.")
            return False
        if not min_str_limit <= len(text) <= max_str_limit:
            logger.error(f"The element in List[str] length not in [{min_str_limit}, {max_str_limit}]")
            return False
    return True


def validate_list_list_str(texts,
                           length_limit: List[int],
                           inner_length_limit: List[int],
                           str_limit: List[int]):
    """
    用于List[List[str]]类型的数据校验
    Args:
        texts: 输入数据字符串列表
        length_limit: 列表长度范围
        inner_length_limit: 内部列表长度范围
        str_limit: 字符串长度范围

    Returns:

    """
    if not isinstance(texts, List):
        logger.error("input is not type List[str]")
        return False
    if len(length_limit) != 2:
        logger.error("the length limit length must equal two")
        return False
    min_length_limit = length_limit[0]
    max_length_limit = length_limit[1]
    if not min_length_limit <= len(texts) <= max_length_limit:
        logger.error(f"The List[List[str]] length not in [{min_length_limit}, {max_length_limit}]")
        return False
    for text in texts:
        res = validate_list_str(text, inner_length_limit, str_limit)
        if not res:
            return False
    return True


def check_db_file_limit(db_path: str, limit: int = DB_FILE_LIMIT):
    """
    检查db文件大小不超过限制limit
    Args:
        db_path: db文件路径
        limit: 大小限制
    """
    if not os.path.exists(db_path):
        return
    if os.path.getsize(db_path) > limit:
        raise Exception(f"The db file '{db_path}' size exceed limit {limit}, failed to add.")


def check_header(headers):
    """
    安全检查headers
    Args:
        headers: headers列表
    """
    if not isinstance(headers, dict):
        logger.error("input is not type dict")
        return False
    if len(headers) > 100:
        logger.error("the length of headers exceed 100")
        return False
    for k, v in headers.items():
        if not isinstance(k, str) or not isinstance(v, str):
            logger.error("The headers is not of the Dict[str, str] type")
            return False
        if len(k) > 100 or len(v) > 1000:
            logger.error("The length of key in headers exceed 100 or the length of value in headers exceed 1000")
            return False
        if v.lower().find("%0d") != -1 or v.lower().find("%0a") != -1 or v.find("\n") != -1:
            logger.error("The headers cannot contain %0d or %0a or \\n")
            return False
    return True


def validate_sequence(param: Union[str, dict, list, tuple, set],
                      max_str_length: int = 1024,
                      max_sequence_length: int = 1024,
                      max_check_depth: int = 1,
                      current_depth: int = 0) -> bool:
    """
    递归校验序列值是否超过允许范围
    Args:
        param: 序列
        max_str_length: int 序列中字符串最大限制
        max_sequence_length: int 序列最大长度限制
        max_check_depth: int 序列校验深度
        current_depth: int 用于计算
    """
    if max_check_depth < 0:
        logger.error(f"sequence nested depth cannot exceed {current_depth - 1}")
        return False

    def check_str(data):
        if not isinstance(data, str):
            return True

        if not 0 <= len(data) <= max_str_length:
            logger.error(f"the {current_depth}th layer string param length must in range[0, {max_str_length}]")
            return False

        return True

    def check_dict(data):
        for k, v in data.items():
            if not (check_str(k) and validate_sequence(v, max_str_length, max_sequence_length, max_check_depth - 1,
                                                       current_depth + 1)):
                return False

        return True

    def check_list_tuple_set(data):
        for item in data:
            if not validate_sequence(item, max_str_length, max_sequence_length, max_check_depth - 1,
                                     current_depth + 1):
                return False

        return True

    if not isinstance(param, (str, dict, set, list, tuple)):
        return True

    if isinstance(param, str):
        return check_str(param)

    if not 0 <= len(param) <= max_sequence_length:
        logger.error(f"the {current_depth}th layer param length must in range[0, {max_sequence_length}]")
        return False

    if isinstance(param, (set, list, tuple)):
        return check_list_tuple_set(param)

    if isinstance(param, dict):
        return check_dict(param)

    return True


def validate_lock(lock) -> bool:
    return isinstance(lock, (multiprocessing.synchronize.Lock, _thread.LockType))


def check_pathlib_path(path) -> bool:
    try:
        if not isinstance(path, pathlib.Path):
            logger.error("param not type pathlib.Path")
            return False
        FileCheck.check_input_path_valid(path.as_posix(), check_blacklist=True)
        FileCheck.check_filename_valid(path.as_posix(), max_length=MAX_SQLITE_FILE_NAME_LEN)
        return True
    except (TypeError, ValueError) as e:
        logger.error(f"input path is illegal, exception: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return False


def get_lang_param(input_param: dict) -> str:
    if "lang" in input_param:
        if not isinstance(input_param.get("lang"), str):
            raise KeyError("lang param error, it should be str type")
        if input_param.get("lang") not in ["zh", "en"]:
            raise ValueError(f"lang param error, value must be in [zh, en]")
    return input_param.get("lang", "zh")


def check_embed_func(embed_func) -> bool:
    if isinstance(embed_func, dict):
        if len(embed_func) > 2:
            logger.error("only support dense and sparse key in embed_func.")
            return False
        if set(embed_func.keys()).difference({"dense", "sparse"}):
            logger.error("only support dense and sparse key in embed_func.")
            return False
        all_none_flag = True
        for key in embed_func.keys():
            if isinstance(embed_func.get(key), Callable):
                all_none_flag = False
            elif embed_func.get(key) is None:
                continue
            else:
                logger.error(EMBED_FUNC_TIP)
                return False
        if all_none_flag:
            logger.error(EMBED_FUNC_TIP)
            return False
        return True
    elif isinstance(embed_func, Callable):
        return True
    else:
        logger.error(EMBED_FUNC_TIP)
        return False


def validate_embeddings(embeddings: Any) -> Tuple[bool, str]:
    """
    Validates the structure and type of embedding data.

    Args:
        embeddings: The embedding data to validate (List[List[float]] or List[Dict[int, float]]).

    Returns:
        Tuple[bool, str]: (True, "") if valid, (False, error_msg) if invalid.
    """
    if not isinstance(embeddings, list):
        return False, f"Embeddings must be a list, but got {type(embeddings)}."
    if not embeddings:
        return False, "Embeddings cannot be empty list"

    if not (0 < len(embeddings) <= MAX_EMBEDDINGS_SIZE):
        return False, "Embeddings size must be in range [1, 1024*1024]"

    # Find first non-empty element to determine type
    first_element = next((x for x in embeddings if x), None)
    if first_element is None:
        return False, "Embeddings cannot consist of empty elements"

    if isinstance(first_element, list):
        # Validate List[List[float]] case
        try:
            _ = np.array(embeddings)
        except ValueError:
            return False, "Embeddings must be convertible to numpy.ndarray"

    elif isinstance(first_element, dict):
        # Validate List[Dict[int, float]] case
        if not all(isinstance(x, dict) for x in embeddings):
            return False, "Embeddings must contain only dictionaries"
        if not all(isinstance(k, int) and isinstance(v, (int, float, np.floating))
                   for x in embeddings for k, v in x.items()):
            return False, "All keys must be ints and values must be float or int values"
    else:
        return False, "Embeddings must be lists of floats or dicts of int to float"

    return True, ""


def _check_sparse_and_dense(vec_ids: List[int], dense: Optional[np.ndarray] = None,
                            sparse: Optional[List[Dict[int, float]]] = None):
    if len(set(vec_ids)) != len(vec_ids):
        raise ValueError("vec_ids contain duplicated value")
    if dense is None and sparse is None:
        raise ValueError("dense and sparse are both None while updating")
    elif dense is None:
        if len(vec_ids) != len(sparse):
            raise ValueError("sparse input lengths mismatch while updating")
    elif sparse is None:
        if len(vec_ids) != len(dense):
            raise ValueError("dense input lengths mismatch while updating")
    elif not len(vec_ids) == len(dense) == len(sparse):
        raise ValueError("dense, sparse and id input lengths mismatch while updating")


def get_model_max_input_length(config):
    position_offset = 0
    try:
        if config.model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = config.pad_token_id + 1

        if hasattr(config, "max_seq_length"):
            max_input_length = config.max_seq_length
        else:
            max_input_length = config.max_position_embeddings - position_offset
    except AttributeError as err:
        logger.error(f"get model config failed because:{err}")
        return 0

    return max_input_length


def write_to_json(file_path: str, data: Union[dict, list], encrypt_fn: Callable = None):
    FileCheck.check_input_path_valid(file_path, check_blacklist=True)
    FileCheck.check_filename_valid(file_path)
    W_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(file_path, W_FLAGS, MODES), 'w') as f:
            if encrypt_fn:
                data = run_and_check_callback(encrypt_fn, json.dumps(data))
                f.write(data)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, JSONDecodeError):
        logger.error(f"Error saving JSON data to {file_path}, invalid json format.")
    except (OSError, IOError) as e:
        logger.error(f"Error saving graph to {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")


def read_graph_file(file_path: str, decrypt_fn: Callable = None):
    FileCheck.check_input_path_valid(file_path, check_blacklist=True)
    FileCheck.check_filename_valid(file_path)
    R_FLAGS = os.O_RDONLY
    MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    with os.fdopen(os.open(file_path, R_FLAGS, MODES), 'r', encoding="utf-8") as f:
        data = f.read()
    if decrypt_fn:
        data = run_and_check_callback(decrypt_fn, data)
    return json.loads(data)


def run_and_check_callback(callback_fun: Callable, input_str: str):
    result = callback_fun(input_str)
    if not isinstance(result, str):
        raise ValueError("the return value of callback function is not str.")
    if len(result) > GRAPH_FILE_LIMIT:
        raise ValueError(f"the length of return value of callback function is too long, exceeding {GRAPH_FILE_LIMIT}.")
    return result
