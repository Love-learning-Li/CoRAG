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
from typing import List, Dict

from loguru import logger
from transformers import PreTrainedTokenizerBase

from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params, validate_list_str, MB
from mx_rag.llm.llm_parameter import LLMParameterConfig

DEFAULT_LLM_TIMEOUT = 10 * 60

SYSTEM_PROMPT = """你是一个问答对生成专家，请按照下列要求，根据给定输入的标题和正文信息，按照输出格式
生成{qas_num_area}个问题，并给出每个问题对应的参考段落，以帮助用户更好地理解文本内容或获取所需信息，请用中文回答。

【要求】
1.**问题生成**：
  -问题应当基于文本内容，客观准确，有意义且有助于理解文本。
  -确保每个问题清晰明了，主谓宾要明确，避免含糊不清的表达，确保问题的内容准确清晰。
2.**答案生成**：
  -为每个问题生成一个准确的答案，确保答案与问题紧密相关，且能够提供文本中对应的信息。
  -答案应当正确、完整且具有可读性，并且涵盖参考段落中的所有要点。
  -答案应提供足够的深度和广度，包含完整详细的信息，长度不要过短，以帮助用户全面理解问题。
3.**质量控制**：
  -确保生成的问答对质量高，答案正确、完整且具有可读性。
  -生成的问答对应当能够满足用户的需求，提供有价值的信息。
4.**多样性**：
  -避免生成重复或相似的问题，保持问题之间的独立性和多样性。
  -确保问题能够覆盖文本的多个方面，提供全面的信息。
5.**格式规范**：
  -按照用户要求或指定的格式，生成清晰明了的问题和对应的答案。
  -确保问答对之间的对应关系清晰，便于用户理解和使用。
6.**上下文理解**：
  -能够理解文本的上下文，包括文本中的隐含信息。
  -根据上下文生成更加丰富和准确的问题和答案。
7.**语言适应性**：
  -根据文本的语言风格和目标受众，调整问题的表达方式。
  -使问题的表达更加自然和易于理解，适应不同用户的需求。

【输出格式】
Q1：如何查询成都火车站的停运列车？
参考段落：'查询方式：铁路12306网页首页。查询流程：第一步：进入铁路12306app首页，点击【车站大屏】；第二步：左上角车站名下拉选择成都东站；第三步：搜索框输入车次即可查询车次情况。'
Q2：四川省将洪水灾害防御响应提升至哪个级别？
参考段落：四川将洪水灾害防御四级响应提升至三级。
Q3：在7月14日，四川省气象台发布了哪种天气预警？
参考段落：7月14日15时30分，四川省气象台继续发布暴雨蓝色预警。
"""

USER_PROMPT = """
【输入数据】
【标题】{title_area}
【正文】{content_area}
"""


class QAGenerationConfig:
    """
    功能描述:
        QA问题对生成参数
    Attributes:
        titles: 文章标题列表
        contents: 文章内容列表
        tokenizer: 预训练的分词器
        llm: 文本到文本的语言模型
        max_tokens: 最大生成token数，默认为1000
        qas_num: 生成的问题数，默认为5
    """

    @validate_params(
        titles=dict(validator=lambda x: validate_list_str(x, [1, 10000], [1, 100]),
                    message="param must meets: Type is List[str], "
                            "list length range [1, 10000], str length range [1, 100]"),
        contents=dict(validator=lambda x: validate_list_str(x, [1, 10000], [1, MB]),
                      message="param must meets: Type is List[str], "
                              f"list length range [1, 10000], str length range [1, {MB}]"),
        tokenizer=dict(validator=lambda x: isinstance(x, PreTrainedTokenizerBase),
                       message="param must be instance of PreTrainedTokenizerBase"),
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM),
                 message="param must be instance of Text2TextLLM"),
        max_tokens=dict(validator=lambda x: isinstance(x, int) and 500 <= x <= 100000,
                        message="param must be int and value range [500, 100000]"),
        qas_num=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= 10,
                     message="param must be int and value range [1, 10]")
    )
    def __init__(self, titles: List[str], contents: List[str], tokenizer: PreTrainedTokenizerBase, llm: Text2TextLLM,
                 max_tokens: int = 1000, qas_num: int = 5):
        self.titles = titles
        self.contents = contents
        self.tokenizer = tokenizer
        self.llm = llm
        self.max_tokens = max_tokens
        self.qas_num = qas_num


class QAGenerate:
    """
    功能描述:
        问答生成类，用于生成问答对
    Attributes:
        config: QAGenerationConfig对象
    """

    @validate_params(
        config=dict(validator=lambda x: isinstance(x, QAGenerationConfig),
                    message="param must be instance of QAGenerationConfig")
    )
    def __init__(self, config: QAGenerationConfig):
        self.config = config

    @staticmethod
    def _generate_qa_from_html(config: QAGenerationConfig, title: str, content: str,
                               system_prompt: str, llm_config: LLMParameterConfig) -> List[str]:
        logger.info(f"LLM generating QA, source title '{title}'")
        title = title.split("-")[0] if len(title.split("-")) > 1 else title
        sys_messages = [{"role": "system", "content": system_prompt}]
        prompt = USER_PROMPT.format(title_area=title, content_area=content)
        result = config.llm.chat(prompt, sys_messages=sys_messages, llm_config=llm_config)
        results = [result.strip()
                   for result in re.split(r'Q\d+[:：]', result)
                   if len(re.findall(r"参考段落[:：]", result)) > 0]
        qas_num = config.qas_num
        if len(results) < qas_num:
            logger.warning(f"The answer does not meet the requirements, skip")
            return []
        # 取出前qas_num个数据
        results = ["".join(result.split("\n")) for result in results[:qas_num]]
        return results

    @staticmethod
    def _split_html_text(text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> str:
        current_tokens_length = len(tokenizer.encode(text))
        if current_tokens_length <= max_tokens:
            return text
        text_lines = [item.strip() for item in text.split("\n") if item.strip()]
        # 句子tokens数量通常长于字符长度，粗略截取接近max_tokens的长度
        for i, text_line in enumerate(text_lines[::-1]):
            if current_tokens_length - len(tokenizer.encode(text_line)) <= max_tokens:
                text_lines = text_lines[:len(text_lines) - i - 1]
                break
            current_tokens_length -= len(text_line)
        else:
            return text_lines[0]
        if len(tokenizer.encode("\n".join(text_lines))) > max_tokens:
            return QAGenerate._split_html_text("\n".join(text_lines), tokenizer, max_tokens)
        return "\n".join(text_lines)

    @validate_params(
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="param must be instance of LLMParameterConfig"))
    def generate_qa(self, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95)) -> Dict:
        if len(self.config.titles) != len(self.config.contents):
            raise ValueError("The length of titles and contents must be equal.")
        final_qas = []
        system_prompt = SYSTEM_PROMPT.format(qas_num_area=str(self.config.qas_num))
        for title, content in zip(self.config.titles, self.config.contents):
            content = QAGenerate._split_html_text(content, self.config.tokenizer, self.config.max_tokens)
            if not content:
                logger.warning(
                    f"The number of tokens in all lines exceeds the max_tokens value {self.config.max_tokens},"
                    f" generate qa of {title} skip")
                continue
            qas = QAGenerate._generate_qa_from_html(self.config, title, content, system_prompt, llm_config)
            if not qas:
                continue
            final_qas.extend(qas)
        qa_pair = {}
        for final_qa in final_qas:
            split_txts = re.split(r"参考段落[:：]", final_qa)
            if len(split_txts) != 2:
                logger.info(f"Can't split '{final_qa}', skip")
                continue
            qa_pair[split_txts[0]] = split_txts[1]
        return qa_pair
