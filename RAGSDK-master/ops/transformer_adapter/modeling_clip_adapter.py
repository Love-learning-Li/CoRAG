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

import io
import math
import os
from typing import Union

import numpy as np
import torch
import torch_npu
import torchair
import torchvision
from cn_clip.clip.model import ResidualAttentionBlock
from loguru import logger
from torch import nn
from torchair.configs.compiler_config import CompilerConfig
from torchvision.transforms import Normalize, Resize, InterpolationMode
from torchvision_npu.datasets._decode_jpeg import extract_jpeg_shape

from cn_clip import clip

old_load_from_name = clip.load_from_name
old_init = ResidualAttentionBlock.__init__
old_attention = ResidualAttentionBlock.attention
enable_clip_speed: bool = True


def read_img(image):
    try:
        f = io.BytesIO()
        image.save(f, format=image.format)
        f.seek(0)
        prefix = f.read(16)
    except AttributeError as e:
        raise ValueError("image object missing required attribute") from e
    except ValueError as e:
        raise ValueError("read image value failed") from e
    except Exception as e:
        raise ValueError("parse image failed") from e

    # DVPP only provides DecodeJpeg op currently
    if prefix[:3] == b"\xff\xd8\xff":
        f.seek(0)
        image_shape = extract_jpeg_shape(f)

        f.seek(0)
        bytes_string = f.read()
        arr = np.frombuffer(bytes_string, dtype=np.uint8)
        uint8_tensor = torch.tensor(arr).npu(non_blocking=True)
        channels = 3

        return torch.ops.torchvision._decode_jpeg_aclnn(
            uint8_tensor, image_shape=image_shape, channels=channels)
    # For other imgae types, use PIL to decode, then convert to npu tensor with NCHW format.
    else:
        img = torch.from_numpy(np.array(image.convert("RGB")))
        img = img.permute((2, 0, 1)).contiguous()
        return img.unsqueeze(0).npu(non_blocking=True)


def new_image_transform(image_size=224):
    def image_processor(image):
        img_tensor = read_img(image)
        rs_img = Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)(
            img_tensor.squeeze(0).float() / 255)

        nl_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
            rs_img)
        return nl_img

    return image_processor


def new_load_from_name(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                       download_root: str = None):
    enable_boost = os.getenv("ENABLE_BOOST", "False")
    if enable_boost not in ("True", "False"):
        raise ValueError("env ENABLE_BOOST value must be True or False")

    if enable_boost == "True" and name.find("ViT") != -1:
        logger.info("enable clip model boost")
        model, preprocess = old_load_from_name(name,
                                               device=device,
                                               download_root=download_root)

        tmp_preprocess = preprocess
        device_name = torch.npu.get_device_name()
        if "910" in device_name:
            torch.ops.torchvision._dvpp_init()
            torchvision.set_image_backend('npu')
            tmp_preprocess = new_image_transform(model.visual.input_resolution)

        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        model.visual = torch.compile(model.visual, dynamic=True, fullgraph=True, backend=npu_backend)

        return model, tmp_preprocess

    else:
        logger.info("disable clip model boost")
        return old_load_from_name(name,
                                  device=device,
                                  download_root=download_root)


def new__init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                use_flash_attention: bool = False):
    enable_boost = os.getenv("ENABLE_BOOST", "False")
    if enable_boost not in ("True", "False"):
        raise ValueError("env ENABLE_BOOST value must be True or False")

    self.boost_flag = enable_boost == "True"

    old_init(self, d_model, n_head, attn_mask)

    if self.boost_flag:
        self.d_model = d_model
        self.n_head = n_head


def new_attention(self, x):
    if not self.boost_flag:
        return old_attention(self, x)

    embed_dim = self.d_model
    if self.n_head == 0:
        raise ValueError("n_head must not be zero")

    head_dim = embed_dim // self.n_head

    if head_dim == 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_head ({self.n_head})")

    seq_len, batch_size, _ = x.shape

    def reshape_to_bsnd(tensor):
        return tensor.permute(1, 0, 2)

    weight = self.attn.in_proj_weight
    bias = self.attn.in_proj_bias
    qkv = nn.functional.linear(x, weight, bias)  # (seq_len, batch, 3 * embed_dim)
    q, k, v = qkv.chunk(3, dim=-1)  # (seq_len, batch, embed_dim)
    q_bsnd = reshape_to_bsnd(q)
    k_bsnd = reshape_to_bsnd(k)
    v_bsnd = reshape_to_bsnd(v)
    attn_output = torch_npu.npu_prompt_flash_attention(
        q_bsnd, k_bsnd, v_bsnd,
        num_heads=self.n_head,
        input_layout="BSH",
        scale_value=1.0 / math.sqrt(head_dim),
        atten_mask=self.attn_mask
    )

    attn_output = attn_output.contiguous().permute(1, 0, 2)
    attn_output = self.attn.out_proj(attn_output)
    return attn_output


ResidualAttentionBlock.__init__ = new__init__
ResidualAttentionBlock.attention = new_attention
clip.load_from_name = new_load_from_name
