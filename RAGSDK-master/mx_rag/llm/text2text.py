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
from typing import List, Optional, Any, Iterator

from pydantic import Field, ConfigDict

from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import safe_get, MB, validate_params, MAX_URL_LENGTH, MAX_MODEL_NAME_LENGTH
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils.url import RequestUtils


def _check_sys_messages(sys_messages) -> bool:
    if sys_messages is None:
        return True

    if not isinstance(sys_messages, list) or len(sys_messages) > 16:
        return False

    for d in sys_messages:
        if not isinstance(d, dict) or len(d) > 16:
            return False
        for k, v in d.items():
            if len(k) > 16 or len(v) > 4 * MB:
                return False
    return True


class Text2TextLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    model_name: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    llm_config: LLMParameterConfig = LLMParameterConfig()
    client_param: ClientParam = ClientParam()

    @property
    def _client(self):
        return RequestUtils(client_param=self.client_param)

    @validate_params(
        query=dict(validator=lambda x: 0 < len(x) <= 4 * MB,
                   message="param length range (0, 4*1024*1024]"),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x),
                          message="param must be None or List[dict], and length of dict <= 16, "
                                  "k-v of dict: len(k) <=16 and len(v) <= 4 * MB"),
        role=dict(validator=lambda x: 1 <= len(x) <= 16, message="param length range [1, 16]"),
        llm_config=dict(validator=lambda x: x is None or isinstance(x, LLMParameterConfig),
                        message="param must be None or LLMParameterConfig")
    )
    def chat(self, query: str,
             sys_messages: Optional[List[dict]] = None,
             role: str = "user",
             llm_config: Optional[LLMParameterConfig] = None):
        ans = ""
        if sys_messages is None:
            sys_messages = []

        if llm_config is None:
            llm_config = self.llm_config
        request_body = self._get_request_body(query, sys_messages, role, llm_config)
        request_body["stream"] = False
        response = self._client.post(url=self.base_url, body=json.dumps(request_body),
                                     headers={"Content-Type": "application/json"})
        if response.success:
            try:
                data = json.loads(response.data)
            except json.JSONDecodeError as e:
                logger.error(f"response content cannot convert to json format: {e}")
                return ans
            except Exception as e:
                logger.error(f"unexpected error while parsing JSON response. Error: {e}")
                return ans

            ans = safe_get(data, ["choices", 0, "message", "content"], "")
            if safe_get(data, ["choices", 0, "finish_reason"], "") == "length":
                logger.info("for the content length reason, it stopped.")
                ans += "......"
        else:
            logger.error("get response failed, please check the server log for details")
        return ans

    @validate_params(
        query=dict(validator=lambda x: 0 < len(x) <= 4 * MB, message="param length range (0, 4*1024*1024]"),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x),
                          message="param must be None or List[dict], and length of dict <= 16, "
                                  "k-v of dict: len(k) <=16 and len(v) <= 4 * MB"),
        role=dict(validator=lambda x: 0 < len(x) <= 16, message="param length range (0, 16]"),
        llm_config=dict(validator=lambda x: x is None or isinstance(x, LLMParameterConfig),
                        message="param must be None or LLMParameterConfig")
    )
    def chat_streamly(self, query: str,
                      sys_messages: Optional[List[dict]] = None,
                      role: str = "user",
                      llm_config: Optional[LLMParameterConfig] = None):
        if sys_messages is None:
            sys_messages = []

        if llm_config is None:
            llm_config = self.llm_config

        request_body = self._get_request_body(query, sys_messages, role, llm_config)
        request_body["stream"] = True
        ans = ""
        response = self._client.post_streamly(url=self.base_url, body=json.dumps(request_body),
                                              headers={"Content-Type": "application/json"})
        for result in response:
            if not result.success:
                logger.error("get response failed")
                break
            chunk = result.data
            if not chunk.strip() or not chunk.startswith(b"data:"):
                continue
            try:
                data = json.loads(chunk[6:].decode("utf-8").strip())
            except json.JSONDecodeError as e:
                break
            except Exception as e:
                logger.error(f"json load error: {e}")
                break

            finish_reason = safe_get(data, ["choices", 0, "finish_reason"], "")
            if finish_reason == "stop":
                break
            elif finish_reason == "length":
                logger.info("for the content length reason, it stopped.")
                ans += "......"
                yield ans
                break
            elif finish_reason == "":
                break
            ans += safe_get(data, ["choices", 0, "delta", "content"], "")
            yield ans

    def _get_request_body(self, query: str, messages: List[dict], role: str, llm_config: LLMParameterConfig):
        messages.append({"role": role, "content": query})
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": llm_config.max_tokens,
            "presence_penalty": llm_config.presence_penalty,
            "frequency_penalty": llm_config.frequency_penalty,
            "seed": llm_config.seed,
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p
        }
        return request_body

    @property
    def _llm_type(self):
        return self.model_name

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.chat(prompt, llm_config=self.llm_config)

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for response in self.chat_streamly(prompt, llm_config=self.llm_config):
            yield GenerationChunk(text=response)
