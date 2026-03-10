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

from langchain_core.embeddings import Embeddings
from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, EMBEDDING_TEXT_COUNT, validate_list_str, \
    STR_MAX_LEN, MAX_URL_LENGTH, MAX_BATCH_SIZE
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


class TEIEmbeddingError(Exception):
    pass


class TEIEmbedding(Embeddings):

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_URL_LENGTH,
                 message=f"param must be str and str length range [1, {MAX_URL_LENGTH}]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"),
        embed_mode=dict(validator=lambda x: isinstance(x, str) and x in ('dense', 'sparse'),
                        message=f"param must be str and in ('dense', 'sparse')"),
    )
    def __init__(self, url: str, client_param=ClientParam(), embed_mode: str = 'dense'):
        self.url = url.rstrip("/")
        self.embed_mode = embed_mode
        self.client = None
        self.headers = {
            'Content-Type': 'application/json'
        }
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

        return TEIEmbedding(**kwargs)

    @staticmethod
    def _process_sparse_data(resp_data):
        res = []
        for sub_list in resp_data:
            if not isinstance(sub_list, list):
                raise TypeError('ech item in tei response must be list')

            data = {}
            for item in sub_list:
                if not isinstance(item, dict) or 'index' not in item or 'value' not in item:
                    raise ValueError('item in tei response must be dict with index and value field')

                data[item['index']] = item['value']
            res.append(data)

        return res

    def _process_data(self, resp_data, texts_batch):
        if not isinstance(resp_data, (list, dict)):
            raise TypeError('tei response is not list or dict')

        data = []
        if self.url.endswith('/embed'):
            if len(resp_data) != len(texts_batch):
                raise ValueError('tei response return data with different size')
            data = resp_data

        elif self.url.endswith('/v1/embeddings'):
            if "data" not in resp_data:
                raise ValueError('tei response has no data field')

            if len(resp_data["data"]) != len(texts_batch):
                raise ValueError('tei response return data with different size')

            for item in resp_data["data"]:
                if "embedding" not in item:
                    raise ValueError('tei response has no embedding field')

                data.append(item["embedding"])

        elif self.url.endswith("/embed_sparse"):
            if len(resp_data) != len(texts_batch):
                raise ValueError('tei response return data with different size')

            data = self._process_sparse_data(resp_data)
        else:
            raise ValueError('url is not supported')

        return data

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, EMBEDDING_TEXT_COUNT], [1, STR_MAX_LEN]),
                   message=f"param must meet: Type is List[str], list length range [1, {EMBEDDING_TEXT_COUNT}], "
                           f"str length range [1, {STR_MAX_LEN}]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param must be int and value range [1, {MAX_BATCH_SIZE}]")
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32) -> List[List[float]]:

        texts_len = len(texts)
        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {}
            if self.url.endswith('/embed') or self.url.endswith('/embed_sparse'):
                request_body = {'inputs': texts_batch, 'truncate': True}
            elif self.url.endswith('/v1/embeddings'):
                request_body = {'input': texts_batch}

            resp = self.client.post(self.url, json.dumps(request_body), headers=self.headers)

            if not resp.success:
                raise TEIEmbeddingError("tei get response failed")

            try:
                resp_data = json.loads(resp.data)

                data = self._process_data(resp_data, texts_batch)
                result.extend(data)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response for batch starting at {start_index}: {json_err}")
                raise TEIEmbeddingError(f"Unable to parse TEI response content: {json_err}") from json_err

            except (TypeError, ValueError) as data_err:
                logger.error(f"Error in TEI response data for batch starting at {start_index}: {data_err}")
                raise TEIEmbeddingError(f"TEI response data error: {data_err}") from data_err

            except Exception as e:
                logger.error(f"Unexpected error while processing batch starting at {start_index}: {e}")
                raise TEIEmbeddingError(
                    f"Failed to process TEI response for batch starting at {start_index}: {e}") from e

        return result

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= STR_MAX_LEN,
                  message="param must be str and length range [1, 128 * 1024 * 1024]")
    )
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]
