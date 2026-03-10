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
MXRAGCache 核心功能类
该类主要是给RAG框架提供数据缓存的能力，包括以下功能
1、缓存实例的构造(get_cache, new_cache)
2、缓存的查询(search)，更新(update)，以及刷新(flush)
3、缓存的级联功能(join)
"""
import os
import re
from typing import Any

from gptcache.core import Cache
from loguru import logger

from mx_rag.cache import CacheConfig, SimilarityCacheConfig
from mx_rag.utils.file_check import SecFileCheck
from mx_rag.utils.common import validate_params, GB, TEXT_MAX_LEN, MAX_QUERY_LENGTH


def _default_dump(data: Any) -> str:
    return data


def _default_load(data: str) -> Any:
    return data


class MxRAGCache:
    # 每条缓存的最大缓存字符数
    cache_limit: int = TEXT_MAX_LEN

    # 最大级联个数
    cache_join_size_limit: int = 6

    # 当前级联个数
    current_join_size: int = 0

    # 是否输出详细日志
    verbose: bool = False

    @validate_params(
        cache_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) < 64 and bool(re.fullmatch(r'[0-9a-zA-Z_]+', x)),
            message="param must meets: Type is str, length range (0, 64), match '[0-9a-zA-Z_]+'"),
        config=dict(validator=lambda x: isinstance(x, CacheConfig) or isinstance(x, SimilarityCacheConfig),
                    message="param must be instance of CacheConfig or SimilarityCacheConfig")
    )
    def __init__(self,
                 cache_name: str,
                 config: CacheConfig):
        self.__cache_obj = Cache()
        self.config = config
        self.cache_name = cache_name

        try:
            from mx_rag.cache.cache_api.cache_init import _init_mxrag_cache

            self.data_save_path = _init_mxrag_cache(self.__cache_obj, cache_name, config)
        except KeyError:
            logger.error("init rag cache failed because key error")
        except Exception:
            logger.error("init rag cache failed")

    @staticmethod
    def _update(
            llm_data, update_cache_func, *args, **kwargs
    ) -> None:
        """When updating cached data, do nothing, because currently only cached queries are processed"""
        from gptcache.adapter.api import _update_cache_callback
        _update_cache_callback(llm_data, update_cache_func, *args, **kwargs)
        return llm_data

    @classmethod
    @validate_params(
        verbose=dict(
            validator=lambda x: isinstance(x, bool),
            message="param value must be bool")
    )
    def set_verbose(cls, verbose: bool):
        cls.verbose = verbose

    @classmethod
    @validate_params(
        cache_limit=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= TEXT_MAX_LEN,
            message="param value range (0, 1000 * 1000]")
    )
    def set_cache_limit(cls, cache_limit: int):
        cls.cache_limit = cache_limit

    def clear(self):
        if "vector_db" in self.data_save_path:
            self.data_save_path["vector_db"].delete_all()
        if "txt_file" in self.data_save_path and os.path.exists(self.data_save_path["txt_file"]):
            os.remove(self.data_save_path["txt_file"])
        if "vector_file" in self.data_save_path and os.path.exists(self.data_save_path["vector_file"]):
            os.remove(self.data_save_path["vector_file"])
        if "sql_file" in self.data_save_path:
            import sqlite3
            conn = sqlite3.connect(self.data_save_path["sql_file"])
            curses = conn.cursor()
            curses.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = curses.fetchall()
            for table in tables:
                table_name = table[0]
                safe_table_name = conn.execute("SELECT quote(?)", (table_name,)).fetchone()[0]
                curses.execute(f"DELETE FROM {safe_table_name};")
                conn.commit()
            conn.commit()

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_QUERY_LENGTH),
    )
    def search(self, query: str):
        """
        MXRAGCache 查询缓存

        Args:
            query: 需要被查询的缓存问题
        Return:
            answer: 如果命中则为缓存问题，未命中则返回None
        """
        if not self.__cache_obj.has_init:
            raise KeyError("cache not init pls init first")

        from gptcache.adapter.api import adapt, _cache_data_converter

        def llm_handle_none(*llm_args, **llm_kwargs) -> None:
            """Do nothing on a cache miss"""
            return None

        answer = adapt(
            llm_handle_none,
            _cache_data_converter,
            self._update,
            prompt=query,
            cache_obj=self.__cache_obj
        )

        if answer is not None:
            self._verbose_log("Hit!")
        else:
            self._verbose_log("Miss!")
        return answer

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_QUERY_LENGTH,
                   message="param must str and char range is (0, 128 * 1024 * 1024]"),
        answer=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                    message="param must str and char range is (0, 1000 * 1000]")
    )
    def update(self, query: str, answer: str):
        """
        MXRAGCache 更新缓存

        Args:
            query: 需要被缓存的用户问题
            answer: 需要被缓存的用户答案
        Return:
            None
        """
        if not self.__cache_obj.has_init:
            raise KeyError("cache not init pls init first")

        if not self._check_limit(answer):
            self._verbose_log("context length is large no caching")
            return

        from gptcache.adapter.api import adapt, _cache_data_converter

        def llm_handle(*llm_args, **llm_kwargs):
            return answer
        adapt(
            llm_handle,
            _cache_data_converter,
            self._update,
            cache_skip=True,
            prompt=query,
            cache_obj=self.__cache_obj
        )
        self._verbose_log("Update!")

    def flush(self):
        """
        MXRAGCache 强制将缓存数据从内存刷新到磁盘

        Return:
            None
        """
        if not self.__cache_obj.has_init:
            raise KeyError("cache not init pls init first")
        if os.path.exists(self.__cache_obj.data_manager.data_path):
            SecFileCheck(self.__cache_obj.data_manager.data_path, 100 * GB).check()
        self.__cache_obj.flush()
        self._verbose_log("Flush!")

    def get_obj(self):
        """
        MXRAGCache 获得gpt缓存示例，用于兼容langchain等 RAG开源框架

        Return:
            gptcache
        """
        if not self.__cache_obj.has_init:
            raise KeyError("cache not init pls init first")

        return self.__cache_obj

    @validate_params(
        next_cache=dict(validator=lambda x: isinstance(x, MxRAGCache)),
    )
    def join(self, next_cache):
        """
        MXRAGCache 缓存级联

        Args:
            next_cache: 下级缓存
        Return:
            None
        """
        if not self.__cache_obj.has_init:
            raise KeyError("cache not init pls init first")

        self._join_check(next_cache)

        self.__cache_obj.next_cache = next_cache.get_obj()
        MxRAGCache.current_join_size = MxRAGCache.current_join_size + 1

    def _join_check(self, next_cache):
        logger.debug(f"cache deepth:{MxRAGCache.current_join_size} cache_limit:{MxRAGCache.cache_join_size_limit}")

        if MxRAGCache.current_join_size >= MxRAGCache.cache_join_size_limit:
            raise OverflowError(
                f"the number of cache join deepth cannot be greater than {MxRAGCache.cache_join_size_limit}")

        loop_cnt: int = 0
        next_cache_obj = next_cache.get_obj()
        while next_cache_obj is not None:
            if loop_cnt >= MxRAGCache.cache_join_size_limit:
                raise OverflowError(
                    f"the number of cache join deepth cannot be greater than {MxRAGCache.cache_join_size_limit}"
                )

            if next_cache_obj == self.__cache_obj:
                raise ValueError("forbidden loop join")
            next_cache_obj = next_cache_obj.next_cache
            loop_cnt = loop_cnt + 1

    def _check_limit(self, input_text: str):
        return True if (len(input_text) < self.cache_limit) else False

    def _verbose_log(self, log_str: str):
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            log_str: 日志信息
        Return:
            None
        """
        if self.verbose:
            logger.info(log_str)
