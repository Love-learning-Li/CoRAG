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

from langchain.prompts import PromptTemplate
from tqdm import tqdm
from loguru import logger

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils.common import validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN, MAX_PROMPT_LENGTH

MAX_TOKENS = 512
GENERATE_QD_PROMPT = """阅读文章，生成一个相关的问题，例如：
文章：气候变化对海洋生态系统造成了严重的影响，其中包括海洋温度上升、海平面上升、酸化等问题。这些变化对海洋生物种群分布、生态圈的稳定性以及渔业等方面都产生了深远影响。在全球变暖的背景下，保护海洋生态系统已经成为当务之急。 
问题：气候变化对海洋生态系统的影响主要体现在哪些方面？
文章：零售业是人工智能应用的另一个重要领域。通过数据分析和机器学习算法，零售商可以更好地了解消费者的购买行为、趋势和偏好。人工智能技术可以帮助零售商优化库存管理、推荐系统、市场营销等方面的工作，提高销售额和客户满意度。
问题：人工智能是如何帮助零售商改善客户体验和销售业绩的？
请仿照样例对以下文章提{question_number}个相关问题：

文章：{doc}

输出格式为以下，按照问题1，问题2...进行编号，冒号后面不要再出现数字编号：
问题1：...
...

"""


@validate_params(
    llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
    doc_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                  message="param must meets: Type is List[str], "
                          f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]"),
    prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_PROMPT_LENGTH,
                message=f"param must be a str and its length meets (0, {MAX_PROMPT_LENGTH}]"),
    question_number=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 20,
                         message="param must meets: Type is int, length range (0, 20]")
)
def generate_qa_embedding_pairs(llm: Text2TextLLM, doc_list: list[str], prompt: str, question_number: int = 1):
    """使用大模型生成问题对"""
    doc_queries = multi_processing(llm, prompt, doc_list, question_number)
    return doc_queries


def multi_processing(llm, prompt, doc_list, question_number):
    logger.info('start to multi process generate qd')

    # 使用 partial 传递固定参数
    make_request_partial = partial(make_request, llm, prompt, question_number)
    with ThreadPoolExecutor() as executor:
        answers = list(
            tqdm(executor.map(make_request_partial, doc_list), total=len(doc_list))
        )
    doc_queries = {}
    for doc, answer in zip(doc_list, answers):
        rs_list = [re.sub(r"^问题\d*[：:]", "", item) for item in answer.split("\n")]
        # 去掉空值
        filter_list = [x for x in rs_list if x]
        doc_queries[doc] = filter_list
    logger.info('end to multi process generate qd')
    return doc_queries


def make_request(llm, prompt_template, question_number, doc_content):
    generate_qd_prompt = PromptTemplate(
        input_variables=["doc", "question_number"],
        template=prompt_template,
    )
    prompt = generate_qd_prompt.format(
        doc=doc_content,
        question_number=question_number
    )
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
