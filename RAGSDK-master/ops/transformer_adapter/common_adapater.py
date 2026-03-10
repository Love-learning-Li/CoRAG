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
import os
from typing import Optional, List, Union, Tuple

import torch
import torch_npu
from file_check import FileCheck


def load_acl_transformer():
    # 获取rag sdk安装目录
    rag_sdk_home_path = os.getenv("RAG_SDK_HOME", "")

    if not 0 < len(rag_sdk_home_path) <= 1024:
        raise ValueError("env RAG_SDK_HOME not be set or length over than 1024")

    lib_path = os.path.join(rag_sdk_home_path, "ops/lib/libatb_torch.so")

    FileCheck.check_path_is_exist_and_valid(lib_path, check_real_path=False)
    # 动态库大小限制到100MB
    FileCheck.check_file_size(lib_path, 100 * 1024 * 1024)
    FileCheck.check_file_owner(lib_path)

    torch.classes.load_library(lib_path)


def init_ascend_operations_boost(self, config):
    if not isinstance(config.num_attention_heads, int) or config.num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be a positive integer")
    self.head_size = config.hidden_size // config.num_attention_heads
    self.head_num = config.num_attention_heads
    if hasattr(config, 'world_size'):
        rank = torch.distributed.get_rank()
        rank_size = torch.distributed.get_world_size()
        self.acl_param = json.dumps({"headNum": self.head_num, "layerNormEps": config.layer_norm_eps,
                                     "dk": self.head_size, "layerNum": config.num_hidden_layers, "rank": rank,
                                     "rankSize": rank_size})
    else:
        self.acl_param = json.dumps({"headNum": self.head_num, "layerNormEps": config.layer_norm_eps,
                                     "dk": self.head_size, "layerNum": config.num_hidden_layers})
    self.max_position_embeddings = config.max_position_embeddings

    self.acl_fa_operation = torch.classes.ModelTorch.ModelTorch("bge_large_FlashAttentionModel")

    self.acl_fa_operation.set_param(self.acl_param)

    self.num_layers = config.num_hidden_layers
    self.hidden_size = config.hidden_size
    self.ascend_weight = []
    self.min_cache = torch.full(
        (self.max_position_embeddings, self.max_position_embeddings),
        torch.finfo(torch.half).min, dtype=torch.half).npu()


def init_ascend_weight_boost(self):
    weights: List = []
    weights = [self.state_dict()["embeddings.word_embeddings.weight"],
               self.state_dict()["embeddings.position_embeddings.weight"],
               self.state_dict()["embeddings.token_type_embeddings.weight"],
               self.state_dict()["embeddings.LayerNorm.weight"],
               self.state_dict()["embeddings.LayerNorm.bias"]
               ]
    for i in range(self.num_layers):
        weights_t = []
        weights_layer = self.encoder.layer[i].state_dict()
        weights_t.append(weights_layer["attention.self.query.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.query.bias"])
        weights_t.append(weights_layer["attention.self.key.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.key.bias"])
        weights_t.append(weights_layer["attention.self.value.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.value.bias"])
        weights_t.append(weights_layer["attention.output.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.output.dense.bias"])
        weights_t.append(weights_layer["attention.output.LayerNorm.weight"])
        weights_t.append(weights_layer["attention.output.LayerNorm.bias"])
        weights_t.append(weights_layer["intermediate.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["intermediate.dense.bias"])
        weights_t.append(weights_layer["output.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["output.dense.bias"])
        weights_t.append(weights_layer["output.LayerNorm.weight"])
        weights_t.append(weights_layer["output.LayerNorm.bias"])
        weights.extend(weights_t)
    self.ascend_weight = weights
    self.acl_fa_operation.set_weight(weights)


def prepare_inputs_for_ascend_boost(self, input_ids, position_ids, token_type_ids, attention_mask=None):
    batch_size, seq_len = input_ids.shape
    position_ids = position_ids.npu()
    token_type_ids = token_type_ids.npu()
    attention_mask = attention_mask.float().half()
    mask = attention_mask.clone()
    # -65504.0 表示fp16最小值
    mask[mask == 0] = -65504.0
    mask[mask == 1] = -0.0
    attention_mask_max = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len, device="npu", dtype=torch.half)
    for i in range(batch_size):
        attention_mask_max[i, :seq_len, :seq_len] = mask[i]
    token_offset_tensor = torch.full((batch_size,), seq_len, device="npu", dtype=torch.int32)
    seq_len_tensor = torch.full((batch_size,), seq_len, device="npu", dtype=torch.int32)

    inputs = [input_ids,
              position_ids,
              token_type_ids,
              attention_mask_max,
              token_offset_tensor,
              seq_len_tensor,
              ] + self.layer_id_list
    return inputs


def execute_ascend_operator_boost(self, input_ids, position_ids, token_type_ids, attention_mask=None):
    batch_size, seq_len = input_ids.shape
    acl_inputs = self.prepare_inputs_for_ascend_boost(input_ids, position_ids, token_type_ids, attention_mask)
    tmp_param = json.dumps(
        {"tokenOffset": [seq_len] * batch_size,
         "seqLen": [seq_len] * batch_size
         })
    acl_model_out = self.acl_fa_operation.execute(acl_inputs, tmp_param)
    acl_hidden_state = acl_model_out[0]
    return acl_hidden_state


def generate_position_ids(input_ids: Optional[torch.Tensor] = None,
                          inputs_embeds: Optional[torch.Tensor] = None,
                          is_roberta: bool = False,
                          past_key_values_length: int = 0,
                          padding_idx: Optional[int] = 0):

    if input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]

    batch_size, seq_length = input_shape

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if not is_roberta:
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        return position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        if input_ids is not None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
            return incremental_indices.long() + padding_idx
        else:
            position_ids = torch.arange(
                padding_idx + 1, seq_length + padding_idx + 1, dtype=torch.long, device=device)
            return position_ids.unsqueeze(0).expand(input_shape)


def generate_token_type_ids(device, input_shape, embeddings, token_type_ids: Optional[torch.Tensor] = None):
    if token_type_ids is not None:
        return token_type_ids

    if hasattr(embeddings, "token_type_ids"):
        batch_size, seq_length = input_shape
        buffered_token_type_ids = embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    return token_type_ids
