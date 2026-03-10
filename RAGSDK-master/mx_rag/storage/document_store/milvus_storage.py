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
from typing import List, Union, Optional, Callable

from loguru import logger
from pymilvus import MilvusClient, DataType, Function, FunctionType

from mx_rag.storage.document_store import MxDocument
from mx_rag.storage.document_store.base_storage import Docstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM, KB, TEXT_MAX_LEN, MAX_TOP_K, validate_list_str, \
    STR_MAX_LEN, BOOL_TYPE_CHECK_TIP, MAX_PAGE_CONTENT


class MilvusDocstore(Docstore):
    @validate_params(
        client=dict(validator=lambda x: isinstance(x, MilvusClient),
                    message="param must be instance of MilvusClient"),
        collection_name=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= MilvusDB.MAX_COLLECTION_NAME_LENGTH,
            message="param must be str and length range (0, 1024]"),
        enable_bm25=dict(validator=lambda x: isinstance(x, bool),
                         message="param must be instance of bool"),
        bm25_k1=dict(validator=lambda x: isinstance(x, (float, int)) and 1.2 <= x <= 2.0,
                     message="param must be be range of [1.2, 2]"),
        bm25_b=dict(validator=lambda x: isinstance(x, (float, int)) and 0 <= x <= 1,
                    message="param must be range of [0, 1]"),
        auto_flush=dict(validator=lambda x: isinstance(x, bool) and 0 <= x <= 1,
                        message=BOOL_TYPE_CHECK_TIP),
        encrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="encrypt_fn must be None or callable function"),
        decrypt_fn=dict(validator=lambda x: x is None or isinstance(x, Callable),
                        message="decrypt_fn must be None or callable function")
    )
    def __init__(self, client: MilvusClient, collection_name: str = "doc_store",
                 enable_bm25=True, bm25_k1: float = 1.2, bm25_b: float = 0.75, auto_flush=True,
                 encrypt_fn: Optional[Callable[[str], str]] = None,
                 decrypt_fn: Optional[Callable[[str], str]] = None):
        self._client = client
        self._collection_name = collection_name
        self._enable_bm25 = enable_bm25
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._auto_flush = auto_flush
        self.encrypt_fn = encrypt_fn
        self.decrypt_fn = decrypt_fn
        if not self._client.has_collection(self._collection_name):
            self._create_collection()
        else:
            logger.warning(f"Collection {self._collection_name} already exists")

    @property
    def client(self):
        return self._client

    @property
    def collection_name(self):
        return self._collection_name

    def drop_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

    @validate_params(
        documents=dict(validator=lambda x: isinstance(x, list) and 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
                         message="param must be List[MxDocument] and length range in (0, 1000 * 1000]"),
        document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0,
                         message="param must greater equal than 0")
    )
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
        data = []
        for doc in documents:
            info = dict(
                page_content=self._encrypt(doc.page_content),
                document_id=document_id,
                document_name=doc.document_name,
                metadata=doc.metadata
            )
            if not self._enable_bm25:
                info["sparse_vector"] = {1: 0.1, 2: 0.3}
            data.append(info)
        res = self.client.insert(collection_name=self.collection_name, data=data)
        if self._auto_flush:
            self.flush()
        logger.info(f"Successfully added {res['insert_count']} documents")
        return list(res["ids"])

    @validate_params(
        chunk_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message="param must greater equal than 0")
    )
    def search(self, chunk_id: int) -> Union[MxDocument, None]:
        res = self.client.get(
            collection_name=self.collection_name,
            ids=chunk_id,
            output_fields=["page_content", "metadata", "document_name"]
        )
        result = None
        if res:
            doc = res[0]
            result = MxDocument(
                page_content=self._decrypt(doc["page_content"]),
                metadata=doc["metadata"],
                document_name=doc["document_name"]
            )
        return result

    @validate_params(
        query=dict(
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
            message=f"param must be str and length range (0, {TEXT_MAX_LEN}]"),
        top_k=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= MAX_TOP_K,
            message="param must be int and must in range (0, 10000]"),
        drop_ratio_search=dict(validator=lambda x: isinstance(x, (float, int)) and 0 <= x < 1,
                               message="param must be range of [0, 1)"),
        filter_dict=dict(validator=lambda x: x is None or isinstance(x, dict),
                         message="param filter_dict must be None or dict"))
    def full_text_search(self, query: str, top_k: int = 3,
                         drop_ratio_search: float = 0.2, filter_dict=None) -> List[MxDocument]:
        if not self._enable_bm25:
            logger.error("MilvusDocstore full_text_search failed due to enable_bm25 is False")
            return []

        search_params = {
            # Proportion of small vector values to ignore during the search
            'params': {'drop_ratio_search': drop_ratio_search},
        }
        self._validate_filter_dict(filter_dict)
        doc_filter = filter_dict.get("document_id", []) if filter_dict else []
        res = self._do_bm25_search(query, top_k, search_params, doc_filter)
        result = []
        if not res:
            return []
        try:
            for item in res[0]:
                item["entity"]["metadata"]["score"] = item["distance"]
                result.append(MxDocument(
                    page_content=self._decrypt(item["entity"]["page_content"]),
                    metadata=item["entity"]["metadata"],
                    document_name=item["entity"]["document_name"]
                ))
            return result
        except json.JSONDecodeError as e:
            logger.error(f"parse data from json format failed!: {e}")
            return []
        except KeyError as e:
            logger.error(f"get result from item failed!: {e}")
            return []
        except Exception as e:
            logger.error(f"exception occurred while full text search: {e}")
            return []

    @validate_params(
        document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message="param must greater equal than 0")
    )
    def delete(self, document_id: int):
        """
        Delete all chunks having document_id `document_id`.
        Args:
            document_id: int

        Returns:

        """
        res = self.client.query(
            collection_name=self.collection_name,
            filter=f"document_id == {document_id}",
            output_fields=["id"]
        )
        ids = [x.get("id") for x in res]
        if ids:
            self.client.delete(self.collection_name, ids)
            if self._auto_flush:
                self.flush()
        return ids

    def get_all_chunk_id(self):
        res = self.client.query(self.collection_name, filter="id == 0 or id != 0", output_fields=["id"])
        return [x.get("id") for x in res]

    def get_all_document_id(self) -> List[int]:
        res = self.client.query(self.collection_name, filter="id == 0 or id != 0", output_fields=["document_id"])
        return list(set([x.get("document_id") for x in res]))

    @validate_params(document_id=dict(validator=lambda x: isinstance(x, int) and x >= 0, message=f"document_id must >= 0"))
    def search_by_document_id(self, document_id: int):
        outputs = self.client.query(
            collection_name=self.collection_name,
            filter=f"document_id == {document_id}",
            output_fields=["page_content", "metadata", "document_name"]
        )
        results = [MxDocument(
            page_content=self._decrypt(output["page_content"]),
            metadata=output["metadata"],
            document_name=output["document_name"]
        ) for output in outputs]
        return results

    @validate_params(
        chunk_ids=dict(validator=lambda x: isinstance(x, list) and 0 < len(x) <= MAX_CHUNKS_NUM,
                       message=f"param value range (0, {MAX_CHUNKS_NUM}]"),
        texts=dict(validator=lambda x: validate_list_str(x, [1, MAX_CHUNKS_NUM], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           f"list length range [1, {MAX_CHUNKS_NUM}], str length range [1, {STR_MAX_LEN}]"),
    )
    def update(self, chunk_ids: List[int], texts: List[str]):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length not the same while calling update function.")
        responses = self.client.get(
            collection_name=self.collection_name,
            ids=chunk_ids
        )
        data = []
        for response, text in zip(responses, texts):
            response["page_content"] = self._encrypt(text)
            data.append(response)
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            if self._auto_flush:
                self.flush()
            logger.info(f"Successfully updated chunk ids {chunk_ids}")
        else:
            logger.warning(f"chunk_ids {chunk_ids} not found in MilvusDocstore")

    def flush(self):
        self.client.refresh_load(collection_name=self.collection_name)

    def _do_bm25_search(self, query, top_k, search_params, doc_filter):
        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query],
            "anns_field": 'sparse_vector',
            "limit": top_k,
            "search_params": search_params,
            "output_fields": ["page_content", "metadata", "document_name"]
        }
        if doc_filter:
            search_kwargs["filter"] = "document_id IN {document_list}"
            search_kwargs["filter_params"] = {"document_list": doc_filter}
        return self.client.search(**search_kwargs)

    def _create_collection(self):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        if self._enable_bm25:
            analyzer_params_built_in = {
                "type": "chinese",
                "filter": ["cnalphanumonly"],
            }
            schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=60 * KB,
                             enable_analyzer=True, analyzer_params=analyzer_params_built_in)
        else:
            schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=60 * KB)
        schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["page_content"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse_vector"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )

        if self._enable_bm25:
            schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        if not self._enable_bm25:
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )
        else:
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                    "bm25_k1": self._bm25_k1,
                    "bm25_b": self._bm25_b
                },
            )
        self.client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )

    def _encrypt(self, text):
        if self.encrypt_fn is not None and not self._enable_bm25:
            result = self.encrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= STR_MAX_LEN:
                return result
            else:
                raise ValueError(f"callback function {self.encrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {STR_MAX_LEN}.")
        else:
            return text

    def _decrypt(self, text):
        if self.decrypt_fn is not None and not self._enable_bm25:
            result = self.decrypt_fn(text)
            if isinstance(result, str) and 0 < len(result) <= MAX_PAGE_CONTENT:
                return result
            else:
                raise ValueError(f"callback function {self.decrypt_fn.__name__} returned invalid result. "
                                 f"Expected: str with length 0 < len <= {MAX_PAGE_CONTENT}.")
        else:
            return text
