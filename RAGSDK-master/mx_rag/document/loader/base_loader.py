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
import os
from abc import ABC
import zipfile
import base64
import psutil

from loguru import logger
from PIL import Image

from mx_rag.llm import Img2TextLLM
from mx_rag.utils import file_check
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP_1024, MAX_IMAGE_PIXELS, MIN_IMAGE_WIDTH, \
    MIN_IMAGE_HEIGHT, MAX_BASE64_SIZE


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_WORD_NUM = 500000
    MAX_FILE_CNT = 1024
    MAX_NESTED_DEPTH = 10

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024, message=STR_TYPE_CHECK_TIP_1024),
    )
    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5
        file_check.SecFileCheck(self.file_path, self.MAX_SIZE).check()

    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                # 检查点1：检查文件个数，文件个数大于预期值时上报异常退出
                file_count = len(zip_ref.infolist())
                if file_count >= self.MAX_FILE_CNT * self.multi_size:
                    logger.error(f'zip file ({self.file_path}) contains {file_count} files, exceed '
                                 f'the limit of {self.MAX_FILE_CNT * self.multi_size}')
                    return True
                # 检查点2：检查第一层解压文件总大小，总大小超过设定的上限值
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                    logger.error(f"zip file '{self.file_path}' uncompressed size is {total_uncompressed_size} bytes"
                                 f"exceeds the limit of {self.MAX_SIZE * self.multi_size} bytes, Potential ZIP bomb")
                    return True

                # 检查点3：检查第一层解压文件总大小，磁盘剩余空间-文件总大小<200M
                remain_size = psutil.disk_usage(os.getcwd()).free
                if remain_size - total_uncompressed_size < self.MAX_SIZE * 2:
                    logger.error(f'zip file ({self.file_path}) uncompressed size is {total_uncompressed_size} bytes'
                                 f' only {remain_size} bytes of disk space available')
                    return True

            return False
        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid zip file or is corrupted: {e}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred while checking zip bomb: {e}")
            return True

    def _verify_image_size(self, image_bytes):
        """Verify if the image dimensions are within acceptable limits."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                total_pixels = width * height
                if total_pixels > MAX_IMAGE_PIXELS:
                    logger.warning(f"Image too large: {width}x{height} pixels. Skipping.")
                    return False
                elif width < MIN_IMAGE_WIDTH and height < MIN_IMAGE_HEIGHT:
                    logger.warning(f"Image too small: {width}x{height} pixels. Skipping.")
                    return False

                return True
        except OSError as err:
            logger.warning(f"Failed to open image file: {err}")
            return False
        except ValueError as err:
            logger.warning(f"Invalid image data: {err}")
            return False
        except MemoryError as err:
            logger.warning(f"Insufficient memory to process image: {err}")
            return False
        except Exception as err:
            logger.warning(f"Failed to verify image size: {err}")
            return False

    def _convert_to_base64(self, image_data,
                           max_base64_size=MAX_BASE64_SIZE,  # base64最大长度
                           max_iterations=10):
        """
        将图片转为Base64编码，并控制大小不超过 max_base64_size。
        - 如果原始数据直接符合要求，直接返回；
        - 否则按比例缩小分辨率，直到符合大小要求。
        """
        raw_b64_len = len(image_data) * 4 // 3
        if raw_b64_len <= max_base64_size:
            return base64.b64encode(image_data).decode("utf-8")

        with Image.open(io.BytesIO(image_data)) as image:
            image = image.convert("RGB")

            # 计算缩放比例
            ratio = (max_base64_size / raw_b64_len) ** 0.5
            new_w = int(image.width * ratio)
            new_h = int(image.height * ratio)

            resized_img = image.resize((new_w, new_h), Image.LANCZOS)

            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=90)  # 默认高质量JPEG
            compressed_data = buffer.getvalue()

            return base64.b64encode(compressed_data).decode("utf-8")

    def _interpret_image(self, image_data, vlm: Img2TextLLM):
        img_base64 = self._convert_to_base64(image_data)
        if self._verify_image_size(image_data) is False:
            logger.warning("image size is invalid")
            img_base64, img_summary = "", ""
            return img_base64, img_summary
        # vllm解析图像
        image_url = {"url": f"data:image/jpeg;base64,{img_base64}"}
        image_summary = vlm.chat(image_url=image_url)
        if image_summary == "":
            img_base64 = ""
            logger.warning("image summary func exec failed")
        return img_base64, image_summary
