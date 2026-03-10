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

"""
MXRAGCache 提供的对外API，用于初始化MXRAGCache的参数，在MXRAGCache运行之前进行调用
"""
import os
import re

import cachetools
from gptcache import Cache
from gptcache.manager.scalar_data import CacheBase
from gptcache.similarity_evaluation import ExactMatchEvaluation
from gptcache.config import Config
from gptcache.processor.pre import get_prompt
from loguru import logger

from mx_rag.cache import CacheConfig, SimilarityCacheConfig, EvictPolicy
from mx_rag.cache.cache_similarity.cache_similarity import CacheSimilarity
from mx_rag.cache.cache_storage.cache_vec_storage import CacheVecStorage
from mx_rag.cache.cache_emb.cache_emb import CacheEmb
from mx_rag.utils.common import validate_params, MB, GB
from mx_rag.utils.file_check import SecFileCheck, FileCheck, check_disk_free_space


def _cache_file_secure_check(file_path: str, file_max_size: int):
    # 检查目录是否存在
    if not os.path.exists(file_path):
        logger.warning("cache file not exist will create by flush")
        return

    # 检查已经存在目录的大小是否合法
    SecFileCheck(file_path, file_max_size).check()


def _get_data_save_file(data_save_folder: str, cache_name: str, memory_only: bool = False):
    """
    功能描述:
        内部接口，根据用户配置创建缓存数据目录和文件

    Args:
        data_save_folder:str 缓存存储目录 由用户提供，需要符合安全标准
        cache_name:str cache实例的键值
        memory_only:bool 是否是memory_cache缓存 如果是则只需要创建data_map文件
    Return:
        vector_save_file:str similarity cache向量数据库缓存文件
        sql_save_file:str similarity cache缓存数据缓存文件
        data_save_file:str memory cache的缓存文件
    """
    file_prefix = cache_name

    FileCheck.dir_check(data_save_folder)

    vector_save_file = ""
    sql_save_file = ""
    data_save_file = ""
    if not memory_only:
        vector_save_file = os.path.join(data_save_folder, f"{file_prefix}_vector_cache_file.index")
        _cache_file_secure_check(vector_save_file, 20 * GB)

        sql_save_file = os.path.join(data_save_folder, f"{file_prefix}_sql_cache_file.db")
        _cache_file_secure_check(sql_save_file, 30 * GB)
    else:
        data_save_file = os.path.join(data_save_folder, f"{file_prefix}_data_map.txt")
        _cache_file_secure_check(data_save_file, 100 * MB)

    return vector_save_file, sql_save_file, data_save_file


def _maybe_create_cache_save_folder(config: CacheConfig):
    FileCheck.check_input_path_valid(config.data_save_folder)
    if config.lock:
        with config.lock:
            if not os.path.exists(config.data_save_folder):
                os.makedirs(config.data_save_folder, 0o750)
    else:
        if not os.path.exists(config.data_save_folder):
            os.makedirs(config.data_save_folder, 0o750)

    if check_disk_free_space(os.path.dirname(config.data_save_folder), config.min_free_space):
        raise Exception("Insufficient remaining space, please clear disk space")


def _get_gpt_cache_config(config: CacheConfig) -> Config:
    config = Config(similarity_threshold=config.similarity_threshold,
                    auto_flush=config.auto_flush,
                    disable_report=config.disable_report
                    )
    return config


# 初始化语义近似 cache
def _init_mxrag_similar_cache(cache_obj: Cache, cache_name: str, config: SimilarityCacheConfig):
    """
    功能描述:
        内部接口，根据SimilarityCacheConfig 初始化指定cache_name的cache实例，为语义相关cache

    Args:
        cache_obj:Cache 缓存实例
        cache_name:str cache实例的键值
        config:SimilarityCacheConfig 语义相似缓存配置数据
    Return:
        None
    """
    from gptcache.manager import get_data_manager
    from gptcache.adapter.api import init_similar_cache

    vector_save_file, sql_save_file, _ = _get_data_save_file(config.data_save_folder, cache_name)

    vector_base = CacheVecStorage.create(**config.vector_config, vector_save_file=vector_save_file,
                                         top_k=config.retrieval_top_k)
    cache_base = CacheBase(config.cache_config, sql_url=f'{config.cache_config}:///{sql_save_file}')
    similarity = CacheSimilarity.create(**config.similarity_config)
    embedding = CacheEmb.create(**config.emb_config)

    data_manager = get_data_manager(
        cache_base=cache_base,
        vector_base=vector_base,
        max_size=config.cache_size,
        eviction=config.eviction_policy.value,
        clean_size=config.clean_size
    )

    gptcache_config = _get_gpt_cache_config(config)

    init_similar_cache(
        pre_func=get_prompt,
        cache_obj=cache_obj,
        data_manager=data_manager,
        embedding=embedding,
        evaluation=similarity,
        config=gptcache_config
    )
    return vector_save_file, sql_save_file, vector_base


def _init_mxrag_memory_cache(cache_obj: Cache, cache_name: str, config: CacheConfig):
    """
    功能描述:
        内部接口，根据CacheConfig 初始化指定cache_name的cache实例，为memory_cache only

    Args:
        cache_obj:Cache 缓存实例
        cache_name:str cache实例的键值
        config:CacheConfig memory_cache 缓存配置数据
    Return:
        None
    """
    from gptcache.manager.data_manager import MapDataManager
    from gptcache.adapter.api import init_similar_cache

    _, _, data_save_file = _get_data_save_file(config.data_save_folder, cache_name, True)
    evict_policy_memory_map = {
        EvictPolicy.LRU.value: cachetools.LRUCache,
        EvictPolicy.LFU.value: cachetools.LFUCache,
        EvictPolicy.FIFO.value: cachetools.FIFOCache,
        EvictPolicy.RR.value: cachetools.RRCache
    }
    data_manager = MapDataManager(data_save_file,
                                  config.cache_size,
                                  evict_policy_memory_map.get(config.eviction_policy.value, cachetools.LRUCache))

    gptcache_config = _get_gpt_cache_config(config)

    init_similar_cache(
        pre_func=get_prompt,
        cache_obj=cache_obj,
        data_manager=data_manager,
        embedding=CacheEmb(skip_emb=True),
        evaluation=ExactMatchEvaluation(),
        config=gptcache_config
    )
    return data_save_file


@validate_params(
    cache_name=dict(
        validator=lambda x: isinstance(x, str) and 0 < len(x) < 64 and bool(re.fullmatch(r'[0-9a-zA-Z_]+', x)),
        message="param must meets: Type is str, length range (0, 64), match '[0-9a-zA-Z_]+'"),
)
def _init_mxrag_cache(cache_obj: Cache, cache_name: str, config):
    """
    功能描述:
        内部接口，根据config 初始化指定cache_name的cache实例

    Args:
        cache_obj:Cache 缓存实例
        cache_name:str cache实例的键值
        config:CacheConfig/SimilarityCacheConfig 缓存配置数据
    Return:
        None
    Raises:
        ValueError: 当配置数据不在有效范围内时
    """
    _maybe_create_cache_save_folder(config)
    save_data_path = {}

    if config.config_type == "similarity_cache_config":
        save_data_path["vector_file"], save_data_path["sql_file"], save_data_path["vector_db"] =\
            _init_mxrag_similar_cache(cache_obj, cache_name, config)
    elif config.config_type == "memory_cache_config":
        save_data_path["txt_file"] = _init_mxrag_memory_cache(cache_obj, cache_name, config)
    else:
        logger.error("config type not support. ")
    return save_data_path
