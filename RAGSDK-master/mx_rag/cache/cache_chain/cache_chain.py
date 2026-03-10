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

from typing import Union, Dict, Iterator, Callable
import json
from loguru import logger

from mx_rag.chain import Chain
from mx_rag.cache import MxRAGCache
from mx_rag.utils.common import validate_params, MAX_QUERY_LENGTH, validate_sequence
from mx_rag.llm.llm_parameter import LLMParameterConfig


class CacheChainChat(Chain):
    """
    功能描述:
        适配cache的chain 对用户提供chain和cache的能力，当cache无法命中时，访问大模型
        更新cache

    Attributes:
        _cache: RAGCache
        _chain: 同大模型对话的模块
    """

    @validate_params(
        cache=dict(validator=lambda x: isinstance(x, MxRAGCache), message="param must be instance of MxRAGCache"),
        chain=dict(validator=lambda x: isinstance(x, Chain), message="param must be instance of Chain"),
        convert_data_to_cache=dict(validator=lambda x: isinstance(x, Callable) or x is None,
                                   message="param must be callable function or None"),
        convert_data_to_user=dict(validator=lambda x: isinstance(x, Callable) or x is None,
                                  message="param must be callable function or None")
    )
    def __init__(self,
                 cache: MxRAGCache,
                 chain: Chain,
                 convert_data_to_cache=None,
                 convert_data_to_user=None):
        self._cache = cache
        self._chain = chain
        self._convert_data_to_cache = convert_data_to_cache
        self._convert_data_to_user = convert_data_to_user

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_QUERY_LENGTH,
                  message=f"param length range (0, {MAX_QUERY_LENGTH}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig) or x is None,
                        message="param must be None or LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
        """
        MXRAGCache 根据verbose标志 用于表示是否记录日志。

        Args:
            llm_config: 大模型参数
            text: 用户问题
        Return:
            ans: 用户答案
        """
        cache_ans = self._cache.search(query=text)
        # 缓存存入为什么格式返回什么格式，可能不是json格式的
        if cache_ans is not None:
            try:
                answer = json.loads(cache_ans)
                if answer.get("query"):
                    answer["query"] = text
                return self._data_to_user(answer)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return cache_ans
            except TypeError as e:
                logger.error(f"Type error: {e}")
                return cache_ans
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return cache_ans

        ans = self._chain.query(text, llm_config)

        result = ans
        # 如果是 stream对象需要通过迭代的方式把内容都读取完才能cache
        if isinstance(ans, Iterator):
            result = None
            for res in ans:
                result = res
        try:
            res = json.dumps(self._data_to_cache(result))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON:{e}")
            return result
        except Exception as e:
            logger.error(f"Unexpected error:{e}")
            return result

        self._cache.update(query=text, answer=res)
        return result

    def _data_to_cache(self, data):
        if self._convert_data_to_cache is not None:
            result = self._convert_data_to_cache(data)
            return result

        else:
            return data

    def _data_to_user(self, data):
        if self._convert_data_to_user is not None:
            result = self._convert_data_to_user(data)
            return result

        else:
            return data
