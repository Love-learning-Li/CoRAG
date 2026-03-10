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

import os
from typing import Optional, List, Union, Tuple

import torch
from loguru import logger
from transformers.models.bert.modeling_bert import BertModel, BertConfig, BaseModelOutputWithPoolingAndCrossAttentions

from common_adapater import load_acl_transformer, init_ascend_operations_boost, init_ascend_weight_boost, \
    prepare_inputs_for_ascend_boost, execute_ascend_operator_boost, generate_token_type_ids, generate_position_ids

enable_bert_speed: bool = True

load_acl_transformer()

old_init = BertModel.__init__
old_forward = BertModel.forward


def new__init__(self, config: BertConfig, add_pooling_layer: bool = True):
    enable_boost = os.getenv("ENABLE_BOOST", "False")
    if enable_boost not in ("True", "False"):
        raise ValueError("env ENABLE_BOOST value must be True or False")

    self.boost_flag = enable_boost == "True"

    self.max_seq_len = config.max_position_embeddings
    self.padding_idx = config.pad_token_id
    if self.boost_flag:
        logger.info("enable bert model boost")
        old_init(self, config, add_pooling_layer)
        self.init_ascend_operations_boost(config)
        self.layer_id_list = [torch.tensor([i], dtype=torch.int32).npu() for i in range(config.num_hidden_layers)]
    else:
        logger.info("disable bert model boost")
        old_init(self, config, add_pooling_layer)


def forward_boost(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    if not self.boost_flag:
        return old_forward(self,
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           past_key_values=past_key_values,
                           use_cache=use_cache,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=return_dict)

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).long()
    else:
        position_ids = generate_position_ids(input_ids, inputs_embeds, past_key_values_length=past_key_values_length)

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

    token_type_ids = generate_token_type_ids(device, input_shape, self.embeddings, token_type_ids)

    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    # add acl model
    if not self.ascend_weight:
        self.init_ascend_weight_boost()

    hidden_states = self.execute_ascend_operator_boost(input_ids,
                                                       position_ids,
                                                       token_type_ids,
                                                       attention_mask)

    sequence_output = hidden_states
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
    )


BertModel.__init__ = new__init__
BertModel.init_ascend_operations_boost = init_ascend_operations_boost
BertModel.init_ascend_weight_boost = init_ascend_weight_boost
BertModel.prepare_inputs_for_ascend_boost = prepare_inputs_for_ascend_boost
BertModel.execute_ascend_operator_boost = execute_ascend_operator_boost
BertModel.forward = forward_boost
