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

from typing import List, Optional, Dict

import numpy as np

from mx_rag.reranker.reranker import Reranker
from mx_rag.storage.vectorstore import VectorStore


class MockerReranker(Reranker):
    def rerank(self, query: str, texts: List[str], batch_size: int = 1):
        return [1] if query.lower() == texts[0].lower() else [0]


class MockerVecStorage(VectorStore):
    def update(self, vec_ids: List[int], dense: Optional[np.ndarray] = None,
               sparse: Optional[List[Dict[int, float]]] = None):
        pass

    def delete(self, ids):
        pass

    def search(self, embeddings, k, filter_dict=None):
        pass

    def add(self, embeddings, ids, document_id):
        pass

    def add_sparse(self, ids, sparse_embeddings):
        pass

    def add_dense_and_sparse(self, ids, dense_embeddings, sparse_embeddings, document_id = 0):
        pass

    def get_ntotal(self) -> int:
        return 10

    def get_all_ids(self) -> List[int]:
        return [0]
