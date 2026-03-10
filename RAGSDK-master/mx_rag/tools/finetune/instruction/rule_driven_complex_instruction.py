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
import os.path
import random
import stat

from loguru import logger

from mx_rag.utils.file_check import FileCheck, SecFileCheck

MAX_FILE_SIZE_100M = 100 * 1024 * 1024
NAME = 'name'


class RuleComplexInstructionRewriter:

    def __init__(self):
        self.init_success = False

        requirement_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'prompt', 'rewrite_requirements_AIGC_SR.json')

        SecFileCheck(requirement_path, MAX_FILE_SIZE_100M).check()
        try:
            R_FLAGS = os.O_RDONLY
            MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            with os.fdopen(os.open(requirement_path, R_FLAGS, MODES), 'r', encoding='utf-8') as out:
                self.requirements = json.load(out)
        except json.JSONDecodeError as json_err:
            logger.error(f"unable to load requirements, find JSONDecodeError: {json_err}")
            return
        except Exception as e:
            logger.error(f"unable to load requirements, find Exception: {e}")
            return

        paraphrase_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'prompt', 'requirement_paraphrase.json')

        SecFileCheck(paraphrase_path, MAX_FILE_SIZE_100M).check()
        try:
            R_FLAGS = os.O_RDONLY
            MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            with os.fdopen(os.open(paraphrase_path, R_FLAGS, MODES), 'r', encoding='utf-8') as out:
                requirement_paraphrase = json.load(out)
                self.org2paraphrase = {}
                for item in requirement_paraphrase:
                    self.org2paraphrase[item['original_requirement']] = item['paraphrase_requirement']
        except json.JSONDecodeError as json_err:
            logger.error(f"unable to load org2paraphrase, find JSONDecodeError: {json_err}")
            return
        except Exception as e:
            logger.error(f"unable to load org2paraphrase, find Exception: {e}")
            return

        self.rewrite_type2aspect = {
            '增加指令子任务': 'SubTaskRequirements',
            '增加回答的限制': 'AnswerContentsRequirements',
            '增加领域知识': 'DomainKnowledgeRequirements',
            '增加指令格式': 'ExtraFormattingRequirements',
            '增加指令要求': 'InstructionRequirements',
            '更改指令语言风格': 'LanguageRequirements'
        }

        self.init_success = True

    @staticmethod
    def _generate_new_instruction(req_list, following_mark, background, original_instruction):
        random.shuffle(req_list)
        requirement_str = ""
        for r_index, req in enumerate(req_list):
            requirement_str += f"{r_index + 1}. {req}\n"
        requirement_str = "#改写要求#\n" + requirement_str.strip() + "\n\n"
        requirement_str = requirement_str.replace('#', following_mark)

        new_instruction_name = "#新指令#"
        new_instruction_name = new_instruction_name.replace('#', following_mark)
        if random.random() < 0.5:
            return background + original_instruction + requirement_str + new_instruction_name
        else:
            return background + requirement_str + original_instruction + new_instruction_name

    def get_rewrite_prompts(self, old_instruction: str, rewrite_type: str):
        if not self.init_success:
            logger.warning("RuleComplexInstructionRewriter init failed")
            return old_instruction

        following_mark = random.choice(['', '*', '#', '$', '@', '##', '**'])

        if random.random() < 0.6:
            value = self.org2paraphrase.get('开头')
            if value is None:
                logger.error("键'开头'不存在")
            background = random.choice(value) + \
                         "最后，#新指令#应该是完全独立的，请不要在#新指令#提及任何与#原指令#或者#改写要求#有关的内容。" \
                         "不要将#改写要求#有关的内容直接转为指令。也不要在#新指令#的内容中提及”#新指令#“这样类似的关键词。\n\n"
        else:
            starting_instruction = [
                "你是一个指令改写专家。你需要根据我的#改写要求#，对我给定的#原指令#进行改写，得到#新指令#。一般而言，#新指令#应该满足以下要求：",
                "在语言上，#新指令#应该清晰，容易被大型语言模型或者人类理解。同时，#新指令#应该是一个具体的、确切的问题。",
                "在内容上，#新指令#应该比#原指令#需要更多的知识、推理才能够被完美得回答。请不要随意扩大问题所涉及的广度来增加问题的复杂性。你在改写的过程中，需要优先通过加深问题的深度来增加#新指令#的复杂度。",
                "#新指令#应该是完全独立的，请不要在#新指令#提及任何与#原指令#或者#改写要求#有关的内容。不要将#改写要求#有关的内容直接转为指令。也不要在#新指令#的内容中提及”#新指令#“这样类似的关键词。",
                "请切记不要在#新指令#中以不同格式重复输出同样或者类似的内容，你要控制指令的长度以及可读性。",
                "#新指令#应该能够用中文或者英文回答，请不要生成涉及除了中文、英文以外的指令。"
            ]
            new_starting_instruction = [random.choice(self.org2paraphrase.get(v)) for v in starting_instruction]
            background = new_starting_instruction[0] + '\n'
            instructions = new_starting_instruction[1:]
            random.shuffle(instructions)
            leading_mark = random.random()
            for index, sub_instruction in enumerate(instructions):
                new_mes = f"{index + 1}. {sub_instruction}\n"
                if leading_mark < 0.5:
                    new_mes = f"- {sub_instruction}\n"
                background += new_mes
            background += '\n'

        background = background.replace('#', following_mark)

        original_instruction = random.choice(["#原指令#\n", "#原指令# "])
        original_instruction = original_instruction.replace('#', following_mark)
        original_instruction += old_instruction + "\n\n"

        req_list = []
        aspect = self.rewrite_type2aspect.get(rewrite_type)
        if aspect is None:
            logger.error("键'%s'不存在", rewrite_type)
            return ""

        instruction = random.choice(self.requirements[aspect]['instructions'])
        if 'parameters' in self.requirements[aspect]:
            input_dict = self._process_params(aspect, instruction)

            if len(input_dict) > 0:
                instruction = instruction.format(**input_dict)
        req_list.append(instruction)

        return self._generate_new_instruction(req_list, following_mark, background, original_instruction)

    def _process_params(self, aspect, instruction):
        input_dict = {}

        for parameter in self.requirements[aspect]['parameters']:
            parameter_marker = "{%s}" % parameter[NAME]
            if parameter_marker not in instruction:
                continue

            values = parameter['values']

            if '#' in parameter[NAME]:
                first_layer_selection = random.choice(list(values.keys()))
                first_layer_parameter_marker = "{%s}" % parameter[NAME].split('.')[0]
                if first_layer_parameter_marker in instruction:
                    input_dict[parameter[NAME].split('#')[0].strip()] = first_layer_selection
                values = values[first_layer_selection]

            input_dict[parameter[NAME]] = random.choice(values)

        return input_dict
