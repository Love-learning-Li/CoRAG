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
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm
from loguru import logger
from langchain.prompts import PromptTemplate

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils.common import validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN, MAX_PROMPT_LENGTH

MAX_TOKENS = 512
SCORING_QD_PROMPT = """您的任务是评估给定问题与文档之间的相关性。相关性评分应该在0到1之间，其中1表示非常相关，0表示不相关。评分应该基于文档内容回答问题的直接程度。

请仔细阅读问题和文档，然后基于以下标准给出一个相关性评分：
- 如果文档直接回答了问题，给出接近1的分数。
- 如果文档与问题相关，但不是直接回答，给出一个介于0和1之间的分数，根据相关程度递减。
- 如果文档与问题不相关，给出0。

例如：
问题：小明昨天吃了什么饭？
文档：小明昨天和朋友出去玩，还聚了餐，吃的海底捞，真是快乐的一天。
因为文档直接回答了问题的内容，因此给出0.99的分数

问题：小红学习成绩怎么样？
文档：小红在班上上课积极，按时完成作业，帮助同学，被老师评为了班级积极分子。
文档中并没有提到小红的学习成绩，只是提到了上课积极，按时完成作业，因此给出0.10的分数

请基于上述标准，为以下问题与文档对给出一个相关性评分，评分分数保留小数点后2位数：
问题: {query}
文档: {doc}

"""


@validate_params(
    llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
    query_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                    message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                            f"str length range [1, {STR_MAX_LEN}]"),
    doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message=f"param must meets: Type is List[str], list length range [1, {TEXT_MAX_LEN}], "
                          f"str length range [1, {STR_MAX_LEN}]"),
    prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]")
)
def llm_preferred(llm: Text2TextLLM, query_list: list[str], doc_list: list[str], prompt: str):
    """大模型打分"""
    if len(query_list) != len(doc_list):
        logger.error(f"llm_preferred's query_list and doc_list has different length")
        return []

    score_list = multi_processing(llm, query_list, doc_list, prompt)

    return score_list


def multi_processing(llm, query_list, doc_list, prompt):
    logger.info('start to multi process LLM preferred')

    # 使用 zip 函数结合列表参数
    params = zip(query_list, doc_list)
    # 使用 partial 传递固定参数
    make_request_partial = partial(make_request, llm, prompt)
    with ThreadPoolExecutor() as executor:
        answers = list(tqdm(executor.map(lambda param: make_request_partial(*param), params), total=len(query_list)))
    # 使用正则表达式提取相关性评分中的小数
    score_list = []
    for answer in answers:
        match = re.search(r"(1(\.0{1,2})?|0(\.\d{1,2})?)", answer)
        score = float(0)
        if match:
            try:
                score = float(match.group())
            except ValueError as e:
                logger.error(f"Value error: {e}")
        score_list.append(score)
    logger.info('end to multi process LLM preferred')
    return score_list


def make_request(llm, prompt_template, query, doc):
    scoring_qd_prompt = PromptTemplate(input_variables=["query", "doc"], template=prompt_template)
    prompt = scoring_qd_prompt.format(query=query, doc=doc)
    llm_config = LLMParameterConfig(max_tokens=MAX_TOKENS)
    try:
        return llm.chat(prompt, llm_config=llm_config)
    except TypeError:
        logger.error("Invalid argument type in llm.chat")
        return ''
    except TimeoutError:
        logger.error("LLM request timed out")
        return ''
    except Exception:
        logger.error("llm chat failed")
        return ''
