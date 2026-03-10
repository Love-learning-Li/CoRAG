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

from typing import Union, Dict, Iterator
from multiprocessing import Process, Value, Lock, Queue

from mx_rag.chain.single_text_to_text import SingleText2TextChain
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils.common import validate_params, TEXT_MAX_LEN, MAX_PROMPT_LENGTH


class ParallelText2TextChain(SingleText2TextChain):
    FIRST_RAG_PROMPT = (
        "根据上述已知信息，简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答"
    )

    NEXT_RAG_PROMPT = (
        "下面是已知信息:"
    )

    @validate_params(
        prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                    message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]")
    )
    def __init__(self, prompt: str = FIRST_RAG_PROMPT, **kwargs):
        super().__init__(prompt=prompt, **kwargs)
        self.prefill_done = Value('i', 0)
        self.prefill_queue = Queue()
        self.lock = Lock()

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                  message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95),
              *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        return self._query(text, llm_config)

    def _query(self,
               question: str,
               llm_config: LLMParameterConfig
               ) -> Union[Dict, Iterator[Dict]]:
        """
            推理和检索并行查询 query
            首先开启prefill 推理检测进程，之后进行检索过程，如果检索完成，prefill未完成则此次回答包含检索内容
            如果prefill完成，检索未完成则此次回答不包含检索内容

        Args:
            question: 用户查询问题
            llm_config: 大模型参数
        Returns:
            用户答案
        """
        # 启动prefill检测进程
        prefill_process = Process(target=self._prefill_process, args=(question, llm_config))
        prefill_process.start()

        # 执行检索
        q_docs = self._retrieve_process(question)

        prefill_is_done = False
        # 检测prefill是否完成
        with self.lock:
            prefill_is_done = True if self.prefill_done.value == 1 else False

        # 如果prefill 已经完成则使用prefill结果
        if prefill_is_done:
            answer = self.prefill_queue.get(block=True, timeout=60)
            answer = answer[0]
        # 否则 走正常推理流程
        else:
            q_with_promp = self._prompt + question + "\n" + self.NEXT_RAG_PROMPT
            if q_docs:
                for doc in q_docs:
                    q_with_promp = q_with_promp + doc.page_content

            if not llm_config.stream:
                answer = self._do_query(q_with_promp, llm_config, question=question, q_docs=q_docs)
            else:
                answer = self._do_stream_query(q_with_promp, llm_config, question=question, q_docs=q_docs)

        prefill_process.join()
        self.prefill_done.value = 0
        return answer

    def _prefill_process(self, text: str, llm_config: LLMParameterConfig):
        """
        执行prefill 检测，如果prefill已经完成就把 prefill_done标志位置为1，并继续返回流式推理结果
        Args:
            text: 用户问题

        Returns:
            流式推理结果
        """
        q_with_promp = self._prompt + text
        answer_interator = self._do_stream_query(q_with_promp, llm_config, question=text)
        result = ""

        for ans in answer_interator:
            result = ans
            if self.prefill_done.value == 0:
                with self.lock:
                    self.prefill_done.value = 1

        self.prefill_queue.put([result])

    def _retrieve_process(self, text: str):
        """
        执行检索，通过检索和reranker
        Args:
            text: 用户问题

        Returns:
            流式推理结果
        """
        docs = []
        if self._retriever is not None:
            docs = self._retriever.invoke(text)

        if self._reranker is not None and len(docs) > 0:
            scores = self._reranker.rerank(text, [doc.page_content for doc in docs])
            docs = self._reranker.rerank_top_k(docs, scores)
        return docs
