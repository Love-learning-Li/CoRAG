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
MXRAGCache 配置功能类
提供对外的配置参数，CacheConfig继承子gptCache的Config类，默认为memory_cache
SimilarityCacheConfig 继承CacheConfig提供 语义相似cache
"""
import os
from enum import Enum
from typing import Dict, Any

from mx_rag.utils.file_check import SecDirCheck
from mx_rag.utils.common import validate_params, \
    validate_sequence, validate_lock, MB, GB, BOOL_TYPE_CHECK_TIP, DICT_TYPE_CHECK_TIP


class EvictPolicy(Enum):
    """
    功能描述:
        缓存替换策略

    Attributes:
        LRU(Least Recently Used):替换最近最少使用的缓存
        LFU(Least Frequently Used):替换最不常使用的缓存
        FIFO(First In First Out):替换最先被调入cache的缓存
        RR(Random Replacement):随机替换缓存
    """
    LRU: str = 'LRU'
    LFU: str = 'LFU'
    FIFO: str = 'FIFO'
    RR: str = 'RR'


class CacheConfig:
    """
    功能描述:
        CacheConfig 继承子gptcache的Config，扩展了gptcache的参数

    Attributes:
        config_type: (str) 表明缓存类型，默认为memory_cache_config
        cache_size: (int) 缓存大小，单位是条
        eviction_policy: (EvictPolicy) 替换策略，包含(LRU, LFU, FIFO, RR)
        min_free_space: (int) 落盘路径 最大剩余磁盘空间大小
        auto_flush: (int) 添加多少次进行自动落盘, 刷新频率
        similarity_threshold: (float) 相似度计算阈值
        disable_report: (bool) 是否开启缓存汇报功能
        lock: (lockable) 多进程或者多线程安全锁
        data_save_folder: (str) 缓存数据存储路径
    """

    DEFAULT_SAVE_FOLDER = os.path.join(os.path.expanduser("~"), "Ascend", "mxRag", "cache_save_folder")

    @validate_params(
        cache_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 100000,
                        message="param must meets: Type is int, length range (0, 100000]"),
        eviction_policy=dict(validator=lambda x: isinstance(x, EvictPolicy),
                             message="param must be instance of EvictPolicy"),
        data_save_folder=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024,
                              message="param must be instance of str and path length range (0, 1024]"),
        min_free_space=dict(validator=lambda x: isinstance(x, int) and 20 * MB <= x <= 100 * GB,
                            message="param must meets: Type is int, value range [20 * MB, 100 * GB]"),
        auto_flush=dict(validator=lambda x: isinstance(x, int) and x > 0,
                        message="param must meets: Type is int, and must greater than zero"),
        similarity_threshold=dict(validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 1.0,
                                  message="param must be float or int and value range [0.0, 1.0]"),
        disable_report=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        lock=dict(
            validator=lambda x: x is None or validate_lock(x),
            message="param must be one of None, multiprocessing.Lock(), threading.Lock()")
    )
    def __init__(self,
                 cache_size: int,
                 eviction_policy: EvictPolicy = EvictPolicy.LRU,
                 data_save_folder: str = DEFAULT_SAVE_FOLDER,
                 min_free_space: int = 1 * GB,
                 auto_flush: int = 20,
                 similarity_threshold: float = 0.8,
                 disable_report: bool = False,
                 lock=None):
        if auto_flush > cache_size:
            raise ValueError(f"auto flush value range is (0, {cache_size}]")

        self.config_type = "memory_cache_config"
        self.cache_size = cache_size
        self.eviction_policy = eviction_policy
        self.data_save_folder = data_save_folder
        self.min_free_space = min_free_space
        self.auto_flush = auto_flush
        self.similarity_threshold = similarity_threshold
        self.disable_report = disable_report
        self.lock = lock
        SecDirCheck(self.data_save_folder, 100 * GB).check()


class SimilarityCacheConfig(CacheConfig):
    """
    功能描述:
        SimilarityCacheConfig 继承自CacheConfig，在CacheConfig基础上扩展了语义相似缓存参数

    Attributes:
        config_type: (str) 表明缓存类型，similarity_cache_config
        vector_config: Dict[str, Any] 向量数据库配置参数
        cache_config: str 缓存数据库配置参数
        emb_config: Dict[str, Any] embedding 配置参数
        similarity_config: Dict[str, Any] 相似度 配置参数
        retrieval_top_k: int 检索时的TOPK参数
        clean_size: int 每次添加满的时候删除的元素个数 1 表示每次删除一个
        **kwargs: 配置基类的参数
    """

    @validate_params(
        retrieval_top_k=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1000,
                             message="param must meets: Type is int, value range (0, 1000]"),
        clean_size=dict(validator=lambda x: isinstance(x, int) and x > 0,
                        message="param must meets: Type is int, value greater than 0"),
        cache_config=dict(validator=lambda x: isinstance(x, str) and x == "sqlite",
                          message="param must be 'sqlite' now"),
        vector_config=dict(validator=lambda x: isinstance(x, dict) and validate_sequence(x, max_check_depth=2),
                           message=DICT_TYPE_CHECK_TIP),
        emb_config=dict(validator=lambda x: isinstance(x, dict) and validate_sequence(x),
                        message=DICT_TYPE_CHECK_TIP),
        similarity_config=dict(validator=lambda x: isinstance(x, dict) and validate_sequence(x),
                               message=DICT_TYPE_CHECK_TIP),
    )
    def __init__(self,
                 vector_config: Dict[str, Any],
                 cache_config: str,
                 emb_config: Dict[str, Any],
                 similarity_config: Dict[str, Any],
                 retrieval_top_k: int = 1,
                 clean_size: int = 1,
                 **kwargs):
        super().__init__(**kwargs)

        if clean_size > self.cache_size:
            raise ValueError(f"clean size value range is (0, {self.cache_size}]")

        self.config_type = "similarity_cache_config"
        self.vector_config = vector_config
        self.cache_config = cache_config
        self.emb_config = emb_config
        self.similarity_config = similarity_config
        self.retrieval_top_k = retrieval_top_k
        self.clean_size = clean_size
