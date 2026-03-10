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


EXTRACT_ENTITY_TEMPLATE = """Extract all entities from the following text. As a guideline, a proper noun is 
generally capitalized. You should definitely extract all names and places.
Return the output as a single comma-separated list, or NONE if there is nothing of note to return.

Only output result without other words.

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
Output: Langchain
END OF EXAMPLE

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Sam.
Output: Langchain, Sam
END OF EXAMPLE

Begin!
{input}
Output:

"""

TEXT_RAG_TEMPLATE = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer as concise and accurate as possible.
        Do NOT repeat the question or output any other words.
        Context: {context} 
        Question: {question} 
        Answer:
"""

FAISS_TRIPLE_TEMPLATE = """Use the following knowledge triplets and your knowledge to answer the question 
as simple as possible.\
    Here are the contexts: {triples}.\n
    Here is the question:{question}."""

LLM_PLAIN_TEMPLATE = """Try to answer the following questions as simple as possible: {question}"""

EVAL_GENERATION_TEMPLATE_CN = """我会给你一个问题，对应的正确答案和一个回答。 \
    根据正确答案判断问题的回答是否正确。判断时不要使用你的知识。如果回答很笼统就判断为错误。如果意思相近可以判断为正确。
    你的输出必须严格为 "Correct" 或 "Incorrect". \n
    问题: {question}\n
    正确答案: {answer}\n
    回答: {response}"""

EVAL_GENERATION_TEMPLATE_EN = """I will give you a Question, a corresponding Ground truth, and an Answer.
    Judge the given Answer correctness according to the Ground truth。Do not use your own knowledge during judgement.
    If the meaning of answer contains the ground truth, you can judge it as "Correct", otherwise "Incorrect".
    Your response should be strictly "Correct" or "Incorrect". \n
    Question: {question}\n
    Ground truth: {answer}\n
    Answer: {response}"""