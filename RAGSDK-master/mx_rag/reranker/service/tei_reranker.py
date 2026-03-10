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


import json
from typing import List

from loguru import logger
import numpy as np

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, MAX_TOP_K, MAX_QUERY_LENGTH, TEXT_MAX_LEN, \
    validate_list_str, STR_MAX_LEN, MAX_URL_LENGTH, MAX_BATCH_SIZE, MB
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


class TEIReranker(Reranker):
    TEXT_MAX_LEN = 1000 * 1000

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_URL_LENGTH,
                 message=f"param must be str and str length range [1, {MAX_URL_LENGTH}]"),
        k=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_TOP_K,
               message=f"param must be int and value range [1, {MAX_TOP_K}]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam")
    )
    def __init__(self, url: str, k: int = 1, client_param=ClientParam()):
        super(TEIReranker, self).__init__(k)
        self.url = url.rstrip("/")
        self.client = None
        self.headers = {'Content-Type': 'application/json'}
        try:
            self.client = RequestUtils(client_param=client_param)
        except FileCheckError as e:
            logger.error(f"tei client file param check failed:{e}")
        except PathNotFileException as e:
            logger.error(f"tei client crt is not a file:{e}")
        except Exception:
            logger.error(f"init tei client failed")

    @staticmethod
    def create(**kwargs):
        if "url" not in kwargs or not isinstance(kwargs.get("url"), str):
            logger.error("url param error. ")
            return None

        return TEIReranker(**kwargs)

    @staticmethod
    def _calculate_score(scores_info, scores_len):
        scores = [0.0] * scores_len
        visited = [False] * scores_len
        for score_json in scores_info:
            idx = score_json[0]
            sco = score_json[1]
            if not isinstance(idx, int):
                raise TypeError('index in tei response is not int value')
            if not isinstance(sco, float):
                raise TypeError('score in tei response it not float value')
            if idx >= scores_len or idx < 0:
                raise IndexError('index in tei response is not within valid range')
            if visited[idx]:
                raise ValueError('index in tei response is repeated')

            visited[idx] = True
            scores[idx] = sco

        return scores

    def _process_data(self, resp_data, scores_len):
        if not isinstance(resp_data, (list, dict)):
            raise TypeError('tei response is not list or dict')

        scores_info = []

        if self.url.endswith('/v1/rerank'):
            if 'results' not in resp_data:
                raise ValueError('tei response has no results field')

            for info in resp_data["results"]:
                if not isinstance(info, dict) or 'index' not in info or 'relevance_score' not in info:
                    raise ValueError('results field must be dict with index and relevance_score field')

                scores_info.append((info.get('index'), info.get('relevance_score')))

        elif self.url.endswith('/rerank'):
            for info in resp_data:
                if not isinstance(info, dict) or 'index' not in info or 'score' not in info:
                    raise ValueError('tei response must be dict with index and value field')

                scores_info.append((info.get('index'), info.get('score')))
        else:
            raise ValueError('url is not supported')

        if len(scores_info) != scores_len:
            raise ValueError('tei response has different data length with request')

        return self._calculate_score(scores_info, scores_len)

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MB,
                   message=f"param length range [1, {MB}"),
        texts=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, MB]),
                   message="param must meet: Type is List[str], "
                           f"list length range [1, {TEXT_MAX_LEN}], str length range [1, f{MB}]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param value range [1, {MAX_BATCH_SIZE}]")
    )
    def rerank(self,
               query: str,
               texts: List[str],
               batch_size: int = 32):
        texts_len = len(texts)
        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {}

            if self.url.endswith('/v1/rerank'):
                request_body = {'query': query, 'documents': texts_batch}
            elif self.url.endswith('/rerank'):
                request_body = {'query': query, 'texts': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=self.headers)
            if resp.success:
                try:
                    resp_data = json.loads(resp.data)

                    scores = self._process_data(resp_data, len(texts_batch))
                    result.extend(scores)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to decode JSON response from API: {json_err}")
                    return np.array([])
                except (TypeError, IndexError, ValueError) as e:
                    logger.error(f"Data processing error: {e}")
                    return np.array([])
                except Exception as e:
                    logger.error(f"Unable to process TEI response content, exception: {e}")
                    return np.array([])
            else:
                logger.error(f"TEI request failed.")
                return np.array([])

        return np.array(result)
