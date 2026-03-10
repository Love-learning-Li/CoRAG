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


import copy
from typing import Union, Iterator, List, Dict, Callable

from langchain_core.documents import Document
from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.llm.text2text import _check_sys_messages
from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, TEXT_MAX_LEN, MB
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain import Chain
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import MAX_PROMPT_LENGTH

DEFAULT_RAG_PROMPT = """根据上述已知信息，简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答"""
TEXT_RAG_TEMPLATE = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer as concise and accurate as possible.
        Do NOT repeat the question or output any other words.
        Context: {context} 
        Question: {question} 
        Answer:
"""


def _user_content_builder(query: str, docs: List[Document], prompt: str) -> str:
    """
       默认的用户输入拼接逻辑。

       参数说明：
       ----------
       query : str
           用户原始提问内容。
           例如：“请根据以下材料总结关键要点。”

       docs : List[Document]
           从检索器（retriever）返回的文档对象列表。
           每个 Document 通常包含：
           - page_content：文档内容文本；
           - metadata：元信息（如来源、标题、分数等）。

       prompt : str
           系统预设提示词（例如 RAG 模板、任务指令）。
           会追加到最后一个文档后面，用于指导模型生成更精确的回答。

       返回：
       -----
       str : 拼接后的完整 prompt 文本，作为大模型输入内容。
       """
    final_prompt = ""
    document_separator: str = "\n\n"
    if len(docs) != 0:
        if prompt != "":
            last_doc = docs[-1]
            last_doc.page_content = (last_doc.page_content
                                     + f"{document_separator}{prompt}")
            docs[-1] = last_doc
        final_prompt = document_separator.join(x.page_content for x in docs)

    if final_prompt != "":
        final_prompt += document_separator

    final_prompt += query
    return final_prompt


def _safe_call_builder(builder: Callable, *args, **kwargs) -> str:
    """
    安全地调用用户自定义的 content builder。
    """
    result = builder(*args, **kwargs)
    if isinstance(result, str) and 0 < len(result) <= 4 * MB:
        return result
    else:
        raise ValueError(f"callback function {builder.__name__} returned invalid result. "
                         f"Expected: str with length 0 < len <= 4MB. fallback to default builder.")


class SingleText2TextChain(Chain):

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever),
                       message="param must be instance of BaseRetriever"),
        reranker=dict(validator=lambda x: isinstance(x, Reranker) or x is None,
                      message="param must be None or instance of Reranker"),
        prompt=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PROMPT_LENGTH,
                    message=f"param must be str and length range [1, {MAX_PROMPT_LENGTH}]"),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x),
                          message="param must be None or List[dict], and length of dict <= 16, "
                                  "k-v of dict: len(k) <=16 and len(v) <= 4 * MB"),
        source=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        user_content_builder=dict(validator=lambda x: isinstance(x, Callable),
                                  message="param must be Callable"),

    )
    def __init__(self, llm: Text2TextLLM,
                 retriever: BaseRetriever,
                 reranker: Reranker = None,
                 prompt: str = DEFAULT_RAG_PROMPT,
                 sys_messages: List[dict] = None,
                 source: bool = True,
                 user_content_builder: Callable = _user_content_builder):
        super().__init__()
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._prompt = prompt
        self._sys_messages = sys_messages
        self._source = source
        self._role: str = "user"
        self._user_content_builder = user_content_builder

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                  message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95),
              *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
        return self._query(text, llm_config)

    def _query(self,
               question: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:

        q_docs = self._retriever.invoke(question)

        if self._reranker is not None and len(q_docs) > 0:
            scores = self._reranker.rerank(question, [doc.page_content for doc in q_docs])
            q_docs = self._reranker.rerank_top_k(q_docs, scores)

        q_with_prompt = _safe_call_builder(self._user_content_builder,
                                           question,
                                           copy.deepcopy(q_docs),
                                           self._prompt)

        if not llm_config.stream:
            return self._do_query(q_with_prompt, llm_config, question=question, q_docs=q_docs)

        return self._do_stream_query(q_with_prompt, llm_config, question=question, q_docs=q_docs)

    def _do_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str, q_docs: List[Document]) \
            -> Dict:
        logger.info("invoke normal query")
        resp = {"query": question, "result": ""}
        if self._source:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in q_docs]

        llm_response = self._llm.chat(query=q_with_prompt, sys_messages=self._sys_messages,
                                      role=self._role, llm_config=llm_config)
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str,
                         q_docs: List[Document] = None) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": question, "result": ""}
        if self._source and q_docs:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in q_docs]

        for response in self._llm.chat_streamly(query=q_with_prompt, sys_messages=self._sys_messages,
                                                role=self._role, llm_config=llm_config):
            resp['result'] = response
            yield resp


class GraphRagText2TextChain(SingleText2TextChain):
    def _query(self,
               question: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:
        contexts = self._retriever.invoke(question)
        if self._reranker is not None and len(contexts) > 0:
            scores = self._reranker.rerank(question, contexts)
            contexts = self._reranker.rerank_top_k(contexts, scores)
        input_context = '\n'.join(contexts) if contexts else ""
        prompt = TEXT_RAG_TEMPLATE.format(context=input_context, question=question)
        if self._llm.llm_config.stream:
            return self._do_stream_query(prompt, llm_config, question, [])
        return self._do_query(prompt, llm_config, question, [])

    def _do_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str, q_docs: List[Document]) \
            -> Dict:
        logger.info("invoke normal query")
        resp = {"query": question, "result": ""}
        llm_response = self._llm.chat(query=q_with_prompt, role=self._role, llm_config=llm_config)
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str,
                         q_docs: List[Document] = None) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": question, "result": ""}
        for response in self._llm.chat_streamly(query=q_with_prompt, role=self._role, llm_config=llm_config):
            resp['result'] = response
            yield resp
