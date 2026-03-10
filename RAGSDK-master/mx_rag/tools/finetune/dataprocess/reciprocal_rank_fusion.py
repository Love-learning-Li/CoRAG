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


# Reciprocal Rank Fusion(RRF)
# 倒数排序融合（RRF）是一种将具有不同相关性指标的多个结果集组合成单个结果集的方法
from mx_rag.utils.common import validate_params, validate_list_list_str, TEXT_MAX_LEN, STR_MAX_LEN

MAX_FUSION_LISTS = 10


@validate_params(
    rank_lists=dict(
        validator=lambda x: validate_list_list_str(x, [1, TEXT_MAX_LEN], [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
        message="param must meets: Type is list[list[str]], "
                f"list length range [1, {TEXT_MAX_LEN}], inner list length range [1, {TEXT_MAX_LEN}], "
                f"str length range [1, {STR_MAX_LEN}]"
    ),
    k=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 100,
           message="param must meets: Type is int, length range (0, 100]")
)
def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60):
    # k是常数平滑因子
    fused_rank = {}

    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list):
            rank_score = 1 / (rank + k)
            if item in fused_rank:
                fused_rank[item] += rank_score
            else:
                fused_rank[item] = rank_score

    # 对最终的融合排名列表进行排序
    def sort_key(key):
        return fused_rank.get(key, 0)

    fused_keys = list(fused_rank.keys())
    fused_keys.sort(reverse=True, key=sort_key)
    return fused_keys
