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
import re

from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, MAX_URL_LENGTH, MAX_MODEL_NAME_LENGTH, MB, MAX_PROMPT_LENGTH
from mx_rag.utils.url import RequestUtils


class Img2ImgMultiModel:

    IMAGE_ITEM = "image"

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_URL_LENGTH,
                 message="param must be str and length range (0, 128]"),
        model_name=dict(validator=lambda x: x is None or isinstance(x, str) and 0 < len(x) <= MAX_MODEL_NAME_LENGTH,
                        message="param must be None or str, and str length range (0, 128]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"),
    )
    def __init__(self, url: str, model_name=None, client_param=ClientParam()):
        self._url = url
        self._model_name = model_name
        self._client = RequestUtils(client_param=client_param)
        self.headers = {'Content-Type': 'application/json'}

    @validate_params(
        prompt=dict(validator=lambda x: 0 < len(x) <= MAX_PROMPT_LENGTH,
                    message="param length range (0, 1 * 1024 * 1024]"),
        image_content=dict(validator=lambda x: 0 < len(x) <= 10 * MB,
                           message="param length range (0, 10 * 1024 * 1024]"),
        size=dict(validator=lambda x: re.compile(r"^\d{1,5}\*\d{1,5}$").match(x) is not None,
                  message=r"param must match '^\d{1,5}\*\d{1,5}$'"),
    )
    def img2img(self, prompt: str, image_content: str, size: str = "512*512") -> dict:
        resp = {"prompt": prompt, "result": ""}

        payload = {
            "prompt": prompt,
            self.IMAGE_ITEM: image_content,
            "size": size,
            "model_name": self._model_name
        }

        response = self._client.post(url=self._url, body=json.dumps(payload), headers=self.headers)
        if not response.success:
            logger.error("request img to generate img failed")
            return resp
        try:
            res = json.loads(response.data)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: response content cannot convert to json format")
            return resp
        except Exception as e:
            logger.error(f"json load error")
            return resp

        if self.IMAGE_ITEM not in res:
            logger.error("request img to generate img failed, the response not contain image")
            return resp

        resp["result"] = res[self.IMAGE_ITEM]

        return resp
