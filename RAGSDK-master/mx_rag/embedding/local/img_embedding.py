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

import base64
import io
import os
from typing import List, Union

import torch
from PIL import Image
from tqdm import tqdm

from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, EMBEDDING_TEXT_COUNT, \
    IMG_EMBEDDING_TEXT_LEN, validate_list_str, MB, GB, EMBEDDING_IMG_COUNT, MAX_BATCH_SIZE
from mx_rag.utils.file_check import SecFileCheck, SecDirCheck, safetensors_check

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. ImageEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. ImageEmbedding will run on cpu.")

_CLIP_MODELS = {
    "ViT-B-16": {
        "checkpoint": "clip_cn_vit-b-16.pt",
        "image_size": 224
    },
    "ViT-L-14": {
        "checkpoint": "clip_cn_vit-l-14.pt",
        "image_size": 224
    },
    "ViT-L-14-336": {
        "checkpoint": "clip_cn_vit-l-14-336.pt",
        "image_size": 336
    },
    "ViT-H-14": {
        "checkpoint": "clip_cn_vit-h-14.pt",
        "image_size": 224},
    "RN50": {
        "checkpoint": "clip_cn_rn50.pt",
        "image_size": 224
    },
}


class ImageEmbedding(Embeddings):
    @validate_params(
        model_name=dict(validator=lambda x: isinstance(x, str) and x in _CLIP_MODELS,
                        message=f"param must be str,supported model: {_CLIP_MODELS.keys()}"),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        model_path=dict(validator=lambda x: isinstance(x, str),
                        message=f"param must be str")
    )
    def __init__(self, model_name: str, model_path: str, dev_id: int = 0):
        self.model_name = model_name
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        # 检查模型文件是否已就绪
        SecFileCheck(os.path.join(self.model_path, _CLIP_MODELS[self.model_name]['checkpoint']), 10 * GB).check()

        self.device = "cpu"
        try:
            if is_torch_npu_available():
                self.device = f'npu:{dev_id}'
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')
        import cn_clip.clip as cnclip
        from cn_clip.clip import load_from_name
        self.model, self.preprocess = load_from_name(self.model_name, self.device, self.model_path)
        self.tokenizer = cnclip
        self.model.eval()

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            logger.error("model_path param error. ")
            return None

        return ImageEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, EMBEDDING_TEXT_COUNT], [1, IMG_EMBEDDING_TEXT_LEN]),
                   message="param must meets: Type is List[str], "
                           "list length range [1, 1000 * 1000], str length range [1, 256]"),

        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param must be int and value valid range is [1, {MAX_BATCH_SIZE}]")
    )
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        result = []
        for start_index in tqdm(range(0, len(texts), batch_size), desc='text embedding ...'):
            batch_texts = texts[start_index:start_index + batch_size]
            encode_texts = self.tokenizer.tokenize(batch_texts, context_length=52).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(encode_texts)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1).cpu().tolist()

            result.extend(text_features)

        return result

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= IMG_EMBEDDING_TEXT_LEN,
                  message=f"param must be str, and length range [1, {IMG_EMBEDDING_TEXT_LEN}]"))
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    @validate_params(
        images=dict(
            validator=lambda x: (isinstance(x, list) and len(x) <= EMBEDDING_IMG_COUNT),
            message=f"param must meets: Type is list, list length range [1, {EMBEDDING_IMG_COUNT}]"),
        batch_size=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param must be int and value range [1, {MAX_BATCH_SIZE}]")
    )
    def embed_images(self, images: Union[List[str], List[Image.Image]], batch_size: int = 32) -> List[List[float]]:
        image_features = []

        for start_idx in tqdm(range(0, len(images), batch_size), desc='image embedding ...'):
            batch_images = images[start_idx: start_idx + batch_size]
            tensors_batch = self._preprocess_images(batch_images)
            tensors_batch = torch.stack(tensors_batch).to(self.device)

            with torch.no_grad():
                batch_image_features = self.model.encode_image(tensors_batch)
                # 归一化
                batch_image_features = batch_image_features / batch_image_features.norm(p=2, dim=-1, keepdim=True)
                image_features.extend(batch_image_features.cpu().tolist())

        if not image_features:
            raise Exception("embedding image failed")

        return image_features

    @validate_params(
        images=dict(
            validator=lambda x: all((isinstance(i, str) for i in x)) or all((isinstance(i, Image.Image) for i in x)),
            message=f"param must meets: all item is str or Image.Image")
    )
    def _preprocess_images(self, images: Union[List[str], List[Image.Image]]) -> List[torch.Tensor]:
        tensors_batch = []

        for image in images:
            if isinstance(image, Image.Image):
                image_bytes = image.tobytes()
                image_size = len(image_bytes)
                if not 1 <= image_size <= 10 * MB:
                    raise ValueError(f"image size out of range, size range is [1, {10 * MB}]")
                tensors_batch.append(self.preprocess(image))
                continue

            if not 1 <= len(image) <= 10 * MB:
                raise ValueError(f"image size out of range, size range is [1, {10 * MB}]")

            try:
                blob = base64.b64decode(image)
                with Image.open(io.BytesIO(blob)) as img:
                    tensors_batch.append(self.preprocess(img))
            except Exception as exe:
                raise ValueError("image preprocess failed") from exe

        return tensors_batch
