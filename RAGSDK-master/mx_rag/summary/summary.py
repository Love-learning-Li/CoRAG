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

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from pydantic import BaseModel, ConfigDict

from langchain_core.prompts import PromptTemplate
from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, MB, MAX_PROMPT_LENGTH
from mx_rag.llm.llm_parameter import LLMParameterConfig

_SUMMARY_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template="""使用简洁的语言提取以下内容的摘要，包含尽可能多的关键信息，输出只包含内容信息，请用中文回答\n\n{text}""")

_MERGE_TEXT_SUMMARY_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template="""使用简洁的语言把下面的多个摘要提炼合并成一个摘要，包含尽可能多的关键信息，输出只包含内容信息，请用中文回答\n\n{text}""")


def _thread_pool_callback(worker):
    worker_exception = worker.exception()
    if worker_exception:
        logger.error(
            "called thread pool executor callback function, worker return exception: {}".format(worker_exception))


class Summary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Text2TextLLM
    llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95)
    __counter: int = 0
    __max_texts_length: int = 1024

    @staticmethod
    def _split_summary_by_threshold(texts: List[str], merge_threshold: int) -> List[Tuple[int, int]]:
        split_indices = []
        start_index = 0
        current_length = 0

        for i, s in enumerate(texts):
            if current_length + len(s) <= merge_threshold:
                current_length += len(s)
            else:
                if i != start_index:
                    split_indices.append((start_index, i - 1))
                    start_index = i
                    current_length = len(s)
                else:
                    split_indices.append((start_index, start_index))
                    start_index = i + 1
                    current_length = 0

        if start_index < len(texts):
            split_indices.append((start_index, len(texts) - 1))

        return split_indices

    @validate_params(
        texts=dict(
            validator=lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) and 0 < sum(len(item) for item in x) <= 1*MB,
            message="param must be List[str], and total length range (0, 1048576]"),
        not_summarize_threshold=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1 * MB, message="param value range (0, 1048576]"),
        prompt=dict(validator=lambda x: isinstance(x, PromptTemplate) and set(x.input_variables) == {"text"} and 0 < len(x.template) <= MAX_PROMPT_LENGTH,
                    message="prompt must like PromptTemplate(input_variables=['text'], "
                            "template='length range (0, 1048576]')")
    )
    def summarize(self, texts: List[str], not_summarize_threshold: int = 30,
                  prompt: PromptTemplate = _SUMMARY_TEMPLATE) -> List[str]:
        if len(texts) > self.__max_texts_length:
            raise ValueError(f"texts can not be greater than {self.__max_texts_length}"
                             f",you can set chunk_size to a larger value")

        with ThreadPoolExecutor() as executor:
            submits = []
            no_summary_texts_set = set()
            for i, text in enumerate(texts):
                if len(text) <= not_summarize_threshold:
                    logger.warning(f"the length of the {i}th text is less than {not_summarize_threshold}. therefore, "
                                   f"the summary is not performed.")
                    no_summary_texts_set.add(i)
                    continue

                thread_pool_exc = executor.submit(self._summarize, text, prompt)
                thread_pool_exc.add_done_callback(_thread_pool_callback)

                submits.append(thread_pool_exc)

        res = ["" for _ in range(len(texts))]

        for i, _ in enumerate(texts):
            if i in no_summary_texts_set:
                res[i] = texts[i]
            else:
                res[i] = submits.pop(0).result()

        return res

    @validate_params(
        texts=dict(
            validator=lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) and 0 < sum(len(item) for item in x) <= 1*MB,
            message="param must be list[str], and all length range in (0, 1048576]"),
        merge_threshold=dict(validator=lambda x: isinstance(x, int) and 1024 <= x <= 1 * MB,
                             message="param value range [1024, 1048576]"),
        not_summarize_threshold=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1 * MB,
                                     message="param value range (0, 1048576]"),
        prompt=dict(validator=lambda x: isinstance(x, PromptTemplate) and set(x.input_variables) == {"text"} and 0 < len(x.template) <= MAX_PROMPT_LENGTH,
                    message="prompt must like PromptTemplate(input_variables=['text'], "
                            "template='length range (0, 1048576]')")
    )
    def merge_text_summarize(self, texts: List[str], merge_threshold: int = 4 * 1024, not_summarize_threshold=30,
                             prompt: PromptTemplate = _MERGE_TEXT_SUMMARY_TEMPLATE) -> str:
        if merge_threshold <= not_summarize_threshold:
            raise ValueError("merge_threshold must bigger than not_summarize_threshold.")
        if len(texts) > self.__max_texts_length:
            raise ValueError(f"texts can not be greater than {self.__max_texts_length}"
                             f",you can set chunk_size to a larger value")
        try:
            if self.__counter >= 10:
                raise RecursionError("Maximum recursion depth reached, you can set merge_threshold to a larger value")

            splits = self._split_summary_by_threshold(texts, merge_threshold)
            res = self.summarize(["\n\n".join(texts[s[0]:s[1] + 1]) for s in splits], not_summarize_threshold, prompt)
            self.__counter += 1
            if len(res) > len(texts):
                raise Exception("sub summary number should less than origin summary number")
            if len(res) == 1:
                return res[0]
            elif len(res) > 1:
                return self.merge_text_summarize(res, merge_threshold, not_summarize_threshold, prompt)
            else:
                raise ValueError("summarize failed, get null content")
        except Exception as err:
            raise ValueError(f"summarize failed: {err}") from err
        finally:
            self.__counter = 0

    def _summarize(self, text: str, prompt: PromptTemplate) -> str:
        return self.llm.chat(prompt.format(text=text), llm_config=self.llm_config)
