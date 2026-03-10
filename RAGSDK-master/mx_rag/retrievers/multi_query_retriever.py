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


import re
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import field_validator, ConfigDict

from mx_rag.llm import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.retrievers.retriever import Retriever
from mx_rag.utils.common import TEXT_MAX_LEN, MAX_PROMPT_LENGTH, validate_params

DEFAULT_QUERY_PROMPT_CH = PromptTemplate(
    input_variables=["question"],
    template="""你是一个人工智能语言模型助理。您的任务是根据用户的原始问题，从不同角度改写生成3个问题。
    请从1开始编号且用中文回答，每个问题用换行符分隔开。下面是一个改写例子：
    样例原始问题：
    你能告诉我关于爱因斯坦相关的信息吗？
    样例改写生成后的3个问题：
    1.爱因斯坦的生平和主要科学成就有哪些？
    2.爱因斯坦在相对论和其他物理学领域有哪些重要贡献？
    3.爱因斯坦的个人生活和他对社会的影响是怎样的？
    需要改写的问题：{question}"""
)


class DefaultOutputParser(BaseOutputParser):
    @staticmethod
    def _is_starting_with_number(query: str):
        return bool(re.match(r'\d.*', query))

    def parse(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            if self._is_starting_with_number(line.strip()):
                lines.append(line)
        return lines


class MultiQueryRetriever(Retriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: Text2TextLLM
    prompt: PromptTemplate = DEFAULT_QUERY_PROMPT_CH
    parser: BaseOutputParser = DefaultOutputParser()
    llm_config: LLMParameterConfig = LLMParameterConfig()

    @field_validator('prompt')
    @classmethod
    def _validate_prompt(cls, prompt):
        if set(prompt.input_variables) != {"question"}:
            raise ValueError('prompt.input_variables must include exactly "question".')
        if not (0 < len(prompt.template) <= MAX_PROMPT_LENGTH):
            raise ValueError(f'prompt.template length must be between 1 and {MAX_PROMPT_LENGTH}.')
        return prompt

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        docs = []

        llm_query = self.prompt.format(question=query)
        llm_response = self.llm.chat(query=llm_query, role="user", llm_config=self.llm_config)
        for sub_query in self.parser.parse(text=str(llm_response)):
            doc = super(MultiQueryRetriever, self)._get_relevant_documents(sub_query)
            docs.extend(doc)

        contents = set()
        new_docs = []
        for doc in docs:
            if doc.page_content not in contents:
                new_docs.append(doc)
                contents.add(doc.page_content)

        return new_docs
