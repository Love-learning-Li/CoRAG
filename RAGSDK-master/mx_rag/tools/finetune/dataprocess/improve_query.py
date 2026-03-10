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
from functools import partial

from tqdm import tqdm
from loguru import logger

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.tools.finetune.instruction.rule_driven_complex_instruction import RuleComplexInstructionRewriter
from mx_rag.utils.common import validate_params, validate_list_str, TEXT_MAX_LEN, STR_MAX_LEN

MAX_TOKENS = 512


@validate_params(
    llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
    old_query_list=dict(validator=lambda x: validate_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                        message="param must meets: Type is List[str], "
                                f"list length range [1, {TEXT_MAX_LEN}], str length range [1, {STR_MAX_LEN}]")
)
def improve_query(llm: Text2TextLLM, old_query_list: list[str]):
    """问题重写"""
    new_query_list = multi_processing(llm, old_query_list)

    return new_query_list


def multi_processing(llm, query_list):
    logger.info('start to multi process improve query')

    # 使用 partial 传递固定参数
    make_request_partial = partial(make_request, llm)
    with ThreadPoolExecutor() as executor:
        answers = list(tqdm(executor.map(make_request_partial, query_list), total=len(query_list)))
    logger.info('end to multi process improve query')
    return answers


def make_request(llm, query):
    rewriter = RuleComplexInstructionRewriter()
    prompt = rewriter.get_rewrite_prompts(query, "更改指令语言风格")
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
