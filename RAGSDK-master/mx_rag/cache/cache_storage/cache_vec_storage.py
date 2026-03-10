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
MXRAGCache 的vector storage 适配器类
"""
from typing import List

import numpy as np
from gptcache.manager.vector_data.base import VectorBase, VectorData

from mx_rag.storage.vectorstore import VectorStore
from mx_rag.storage.vectorstore.vector_storage_factory import VectorStorageFactory
from mx_rag.utils.common import validate_params


class CacheVecStorage(VectorBase):
    """
    功能描述:
        CacheVecStorage 为MXRAG适配gptcache 向量数据库功能的适配器

    Attributes:
        _vec_impl: (vectorstore) 来自MXRAG的vector实例
        _top_k: (int) 检索时的top_k参数
    """

    @validate_params(
        vec_store=dict(validator=lambda x: isinstance(x, VectorStore), message="param must be instance of VectorStore"),
        top_k=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= 1000,
                   message="param must be int and value range [1, 1000]")
    )
    def __init__(self, vec_store: VectorStore, top_k: int = 1):
        self._vec_impl = vec_store
        self._top_k = top_k

    @staticmethod
    def create(**kwargs):
        """
        构造CacheVecStorage的静态方法

        Args:
            kwargs: Dict[str, Any] vector 配置参数
        Return:
            vector 实例
        """
        top_k = kwargs.pop("top_k", 5)
        vector_save_file = kwargs.pop("vector_save_file", "")

        """
        针对npu_faiss 如果vector_save_file之前的文件存在，则表明用户可以直接读取之前的数据
        在cache场景 index的缓存文件名由gptcache分配，因此会覆盖load_local_index auto_save_path(资料需要说明)
        """
        vector_type = kwargs.get("vector_type", "")
        if isinstance(vector_type, str) and vector_type == "npu_faiss_db":
            kwargs["load_local_index"] = vector_save_file
            kwargs["auto_save"] = False  # 由gptcache 调用flush进行刷新 因此自动刷新关闭

        vector_base = VectorStorageFactory.create_storage(**kwargs)
        vector_base = CacheVecStorage(vector_base, top_k=top_k)
        return vector_base

    def mul_add(self, datas: List[VectorData]):
        """
        提供批量添加数据的函数

        Args:
            datas:List[VectorData] 批量添加的数据
        Return:
            None
        """
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))

        np_data = np.array(data_array).astype("float16").reshape(1, -1)
        self._vec_impl.add(id_array, np_data)

    def search(self, data: np.ndarray, top_k: int = -1):
        """
        检索数据

        Args:
            data:np.ndarray 查询的数据(经过embedding)
            top_k:查询时的top_k参数
        Return:
            返回命中的top_k个数据
        """
        top_k = top_k if top_k != -1 else self._top_k

        np_data = np.array(data).astype("float16").reshape(1, -1)
        dist, ids = self._vec_impl.search(np_data.tolist(), top_k)[:2]
        ids = [int(i) for i in ids[0]]
        return list(zip(dist[0], ids))

    def rebuild(self, ids=None) -> bool:
        return True

    def delete(self, ids) -> bool:
        """
        提供删除数据的功能

        Args:
            ids:需要删除的位置
        Return:
            True 删除成功
            False 删除失败
        """
        self._vec_impl.delete(ids)
        return True

    def flush(self):
        """
        提供刷新数据的功能，将数据从内存刷新至磁盘

        Return: None
        """
        user_save_file = self._vec_impl.get_save_file()
        if user_save_file:
            self._vec_impl.save_local()

    def close(self):
        """
        给GPTCache提供的接口，用于关闭vector storage，并刷新数据至磁盘
        """
        self.flush()

    def count(self):
        """
        给GPTCache提供的接口，用于gptcache查询vector storage的数据个数

        Return:
            total:(int) 当前向量数据库存放的个数
        """
        return self._vec_impl.get_ntotal()

    def delete_all(self):
        ids = self._vec_impl.get_all_ids()
        self._vec_impl.delete(ids)
