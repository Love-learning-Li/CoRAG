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
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from json_repair import repair_json
from loguru import logger
from tqdm import tqdm
from langchain_core.documents import Document

from mx_rag.graphrag.prompts.extract_graph import TRIPLE_INSTRUCTIONS_CN, TRIPLE_INSTRUCTIONS_EN
from mx_rag.graphrag.prompts.repair_json import JSON_REPAIR_PROMPT
from mx_rag.graphrag.utils.json_util import (
    extract_json_like_substring,
    fix_entity_event_json_string,
    fix_entity_relation_json_string,
    fix_event_relation_json_string,
    normalize_json_string,
)
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import Lang, validate_params, MAX_PROMPT_LENGTH


def _parse_and_repair_json(
    llm: Text2TextLLM,
    text: str,
    answer_start_token: str = "",
    repair_function: Optional[Callable[[str], str]] = None,
    remove_space: bool = False,
    handle_single_quote: bool = False,
    llm_repair_prompt_template: str = JSON_REPAIR_PROMPT,
) -> List[dict]:
    """
    Efficiently parse and repair a JSON-like string, escalating from local fixes to LLM repair.
    """
    json_text = extract_json_like_substring(text, answer_start_token).strip()
    normalized_json_text = normalize_json_string(
        json_text, remove_space, handle_single_quote
    )
    repaired_json_text = repair_json(normalized_json_text, ensure_ascii=False)

    def try_parse(s: str) -> Optional[List[dict]]:
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
        except TypeError as e:
            logger.warning(f"Type error during parsing: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Value error during parsing: {e}")
            return None
        except Exception:
            return None

    def attempt_repair_and_parse(
        repair_strategy: Callable[[str], str], text
    ) -> Optional[List[dict]]:
        try:
            repaired = repair_strategy(text)
            return try_parse(repaired)
        except TypeError as e:
            logger.warning(f"Type error during repair: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Value error during repair: {e}")
            return None
        except Exception:
            return None

    # Try direct parse
    result = try_parse(json_text)
    if result:
        return result

    if repair_function:
        result = attempt_repair_and_parse(repair_function, json_text)
        if result:
            return result

        result = attempt_repair_and_parse(repair_function, repaired_json_text)
        if result:
            return result

    # Try LLM repair
    result = attempt_repair_and_parse(
        lambda s: llm.chat(llm_repair_prompt_template.format(q=s)), json_text
    )
    if result:
        return result

    logger.warning("All repair attempts failed.")
    return []


def generate_relations_cn(
    llm: Text2TextLLM,
    pad_token: str,
    texts: List[str],
    repair_function: Callable[[str], str],
) -> List[List[dict]]:
    """
    Generalized function to generate a list of relations from the model output (Chinese).
    """
    processed_texts = [text.replace(pad_token, "") for text in texts]
    relations = []
    for text in processed_texts:
        relations.append(_parse_and_repair_json(llm, text, "", repair_function, True, True))
    return relations


def generate_relations_en(llm: Text2TextLLM, texts: List[str]) -> List[List[dict]]:
    """
    Generates a list of entity relation dictionaries from the model output (English).
    """
    return [_parse_and_repair_json(llm, text, "", repair_json) for text in texts]


def _check_triple_instructions(triple_instructions: Optional[dict] = None) -> bool:
    """
    Check if the triple instructions are valid.
    """
    if triple_instructions is None:
        return True
    return isinstance(triple_instructions, dict) and all(
        isinstance(triple_instructions.get(k), str)
        and 1 <= len(triple_instructions.get(k)) <= MAX_PROMPT_LENGTH
        for k in ["entity_relation", "event_entity", "event_relation"]
    )


class LLMRelationExtractor:
    """
    LLMRelationExtractor is a class designed to extract relations and entities from text using a language model (LLM).
    It supports multiple configurations for different extraction tasks and languages.
    Attributes:
        llm (Text2TextLLM): The language model used for text-to-text generation.
        pad_token (str): The token used for padding in the LLM.
        language (Lang): The language setting, defaulting to Chinese (Lang.CH).
        triple_instructions (dict): Instructions for extracting triples based on the language.
        user_prompts (dict): User prompts for different extraction tasks (entity_relation, event_entity, event_relation).
    """

    @validate_params(
        max_workers=dict(
            validator=lambda x: x is None or (isinstance(x, int) and 0 < x <= 512),
            message="param must be none or an integer in [1, 512]",
        )
    )
    @validate_params(
        triple_instructions=dict(
            validator=lambda x: _check_triple_instructions(x),
            message=f"Must be None or a dict with keys: 'entity_relation', 'event_entity', 'event_relation' "
                    f"and each value must be a string with length in [1, {MAX_PROMPT_LENGTH}]",
        )
    )
    def __init__(
        self,
        llm: Text2TextLLM,
        pad_token: str,
        language: Lang = Lang.CH,
        max_workers=None,
        triple_instructions: Optional[dict] = None,
    ):
        self.llm = llm
        self.pad_token = pad_token
        self.language = language
        self.max_workers = max_workers
        triple_instructions = triple_instructions or (
            TRIPLE_INSTRUCTIONS_CN if language == Lang.CH else TRIPLE_INSTRUCTIONS_EN
        )
        self.user_prompts = {
            "entity_relation": triple_instructions.get("entity_relation"),
            "event_entity": triple_instructions.get("event_entity"),
            "event_relation": triple_instructions.get("event_relation"),
        }

    @validate_params(
        docs=dict(
            validator=lambda x: isinstance(x, list)
            and all(isinstance(it, Document) for it in x)
            and 0 < len(x) <= 1000000,
            message="param must be a list of Document elements, length range [1, 1000000]",
        )
    )
    def query(self, docs: List[Document]) -> List[dict]:
        outputs = {key: [] for key in self.user_prompts}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for key, user_prompt in self.user_prompts.items():
                texts = [doc.page_content for doc in docs]
                outputs[key] = list(
                    tqdm(
                        executor.map(
                            lambda text: self.llm.chat(f"{user_prompt}{text}"), texts
                        ),
                        total=len(texts),
                        desc=f"Processing {key}",
                    )
                )

        entity_relations = self._process_relations(
            outputs["entity_relation"], fix_entity_relation_json_string
        )
        event_entity_relations = self._process_relations(
            outputs["event_entity"], fix_entity_event_json_string
        )
        event_relations = self._process_relations(
            outputs["event_relation"], fix_event_relation_json_string
        )

        return [
            {
                "raw_text": doc.page_content,
                "file_id": doc.metadata["source"],
                "entity_relations": entity_relations[i],
                "event_entity_relations": event_entity_relations[i],
                "event_relations": event_relations[i],
                "llm_output_entity_entity": outputs["entity_relation"][i],
                "llm_output_event_entity": outputs["event_entity"][i],
                "llm_output_event_event": outputs["event_relation"][i],
            }
            for i, doc in enumerate(docs)
        ]

    def _process_relations(
        self, outputs: List[str], repair_function: Callable
    ) -> List[List[dict]]:
        if self.language == Lang.CH:
            return generate_relations_cn(
                self.llm, self.pad_token, outputs, repair_function
            )
        return generate_relations_en(self.llm, outputs)
