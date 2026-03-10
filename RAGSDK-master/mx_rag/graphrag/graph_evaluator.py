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
import concurrent.futures
import numpy as np
from typing import Any, List, Optional
from dataclasses import dataclass

from loguru import logger
from tqdm import tqdm

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils.common import validate_params, validate_sequence


def _check_relations(relations):
    if (isinstance(relations, list) and 0 < len(relations) < 50000 and
            validate_sequence(relations, max_str_length=4096, max_check_depth=5)):
        return True
    return False


@dataclass
class ProcessResult:
    """Result of processing a single item in evaluation."""
    len_1: int
    len_2: int
    len_3: int
    len1_incorrect: int
    len2_incorrect: int
    len3_incorrect: int
    len1_more: int
    len2_more: int
    len3_more: int


class GraphEvaluator:
    """
    Facade class for evaluating graph-based information extraction results.
    Handles precision, recall, and F1 calculation for entity, event-entity, and event relations.
    """

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM),
                 message="llm must be an instance of Text2TextLLM"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be an instance of LLMParameterConfig")
    )
    def __init__(self, llm: Text2TextLLM, llm_config: LLMParameterConfig):
        """
        Initialize the GraphEvaluator.

        Args:
            llm (Text2TextLLM): The language model used for evaluation.
            llm_config (LLMParameterConfig): Configuration for the language model.
        """
        self.llm = llm
        self.llm_config = llm_config

    @staticmethod
    def _calculate(length: int, incorrect: int, more: int) -> List[Optional[float]]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            length (int): Number of items.
            incorrect (int): Number of incorrect items.
            more (int): Number of missing items.

        Returns:
            List[Optional[float]]: [precision, recall, f1]
        """
        if length > 0 and (length > incorrect):
            precision = round((length - incorrect) / length, 4)
            recall = round((length - incorrect) / (length - incorrect + more), 4)
        else:
            precision = recall = np.nan

        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            f1 = np.nan

        return [precision, recall, f1]

    @staticmethod
    def _count_origin(entity_relations: Any, event_entity_relations: Any, event_relations: Any) -> List[int]:
        """
        Count the number of original entity, event-entity, and event relations.

        Args:
            entity_relations (Any): Entity relations.
            event_entity_relations (Any): Event-entity relations.
            event_relations (Any): Event relations.

        Returns:
            List[int]: [entity_rel_count, event_entity_count, event_rel_count]
        """
        len_1 = GraphEvaluator._safe_len(entity_relations)
        len_3 = GraphEvaluator._safe_len(event_relations)
        len_2 = 0
        entities = []
        for item in event_entity_relations:
            if isinstance(item, dict) and 'Entity' in item:
                for entity in item['Entity']:
                    entities.append(entity)
        len_2 = GraphEvaluator._safe_len(entities)
        return [len_1, len_2, len_3]

    @staticmethod
    def _remove_empty_lines(text: str) -> str:
        """
        Remove empty lines from a string.

        Args:
            text (str): Input text.

        Returns:
            str: Text without empty lines.
        """
        return "\n".join([line for line in text.split("\n") if line.strip()])

    @staticmethod
    def _extract_entities_from_text(text: str) -> List[str]:
        """
        Extract entities from a string using regex.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of extracted entities.
        """
        pattern = r'\[([^[\]]*)\]'
        match = re.search(pattern, text)
        if not match:
            return []
        entity_match = match.group().strip("[]").split("', '")
        return [i.strip("'") for i in entity_match]

    @staticmethod
    def _safe_len(obj: Any) -> int:
        """
        Safely get the length of an object.

        Args:
            obj (Any): The object to get the length of.

        Returns:
            int: The length or 0 if not possible.
        """
        try:
            return len(obj)
        except TypeError as e:
            logger.warning(f"Type error: Object is not iterable or has no length - {e}")
            return 0
        except ValueError as e:
            logger.warning(f"Value error: Invalid value encountered - {e}")
            return 0
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
            return 0

    def _get_more(self, text: str, entity_relations: Any, event_entity_relations: Any,
                  event_relations: Any) -> List[str]:
        """
        Get more unrecognized relations/entities from the model.

        Args:
            text (str): Original text.
            entity_relations (Any): Entity relation dictionary.
            event_entity_relations (Any): Event-entity relation dictionary.
            event_relations (Any): Event relation dictionary.

        Returns:
            List[str]: [entity_rel_more, event_entity_more, event_rel_more]
        """
        text_prompt = (
            "I need you to help me to find more unrecognized results for information extraction, "
            "The text paragraphs are provided as followed:"
        )
        prompts = [
            "\nIf the relations are all recognized, only output \"All recognized!\", don't output anything else. "
            "Else, output unrecognized triples strictly line by line as the original format like"
            "{'Head': '', 'Relation': '', 'Tail': ''}.Output triples as least as possible.No blank lines."
            "No other words.Line breaks are not allowed in a triple. "
            "Output triples shouldn't contain any given event relationship, or similar to the given event "
            "relationship. The given entity relationship recognition results as follows: ",

            "\nIf the entities are all recognized, output \"All recognized!\" only, don't output anything else. "
            "Else, output unrecognized triples strictly line by line as the original format like"
            "{'Event': 'sentence', 'Entity': ['entity1','entity2','...']}.Line breaks are not allowed in a triple."
            "Output triples as least as possible. Given entities or entities similar to them should be excluded "
            "from the entity list. No blank lines.No other words. The given entity recognition results as "
            "followed: ",

            "\nIf the relations are all recognized, output \"All recognized!\" only, don't output anything else. "
            "Else, output unrecognized triples strictly line by line as the original format like"
            "{'Head': '', 'Relation': '', 'Tail': ''}. The relationships types are :before, after, at the same "
            "time, because, and as a result. No blank lines.No other words.Line breaks are not allowed in a "
            "triple.Output triples as least as possible. Output triples shouldn't contain any given event "
            "relationship, or similar to the given event relationship. The given event relationship recognition "
            "results as followed:"
        ]
        return [
            self.llm.chat(f"{text_prompt}{text}{prompts[0]}{entity_relations}", llm_config=self.llm_config),
            self.llm.chat(f"{text_prompt}{text}{prompts[1]}{event_entity_relations}", llm_config=self.llm_config),
            self.llm.chat(f"{text_prompt}{text}{prompts[2]}{event_relations}", llm_config=self.llm_config)
        ]

    def _count_more(self, task1_more_answer: str, task2_more_answer: str, task3_more_answer: str) -> List[int]:
        """
        Count the number of additional (unrecognized) relations/entities.

        Args:
            task1_more_answer (str): Entity relation more answer.
            task2_more_answer (str): Event-entity relation more answer.
            task3_more_answer (str): Event relation more answer.

        Returns:
            List[int]: [entity_rel_more_count, event_entity_more_count, event_rel_more_count]
        """
        task1_more_answer = self._remove_empty_lines(task1_more_answer)
        task2_more_answer = self._remove_empty_lines(task2_more_answer)
        task3_more_answer = self._remove_empty_lines(task3_more_answer)

        len1_more = 0 if "All recognized" in str(task1_more_answer) else len(task1_more_answer.split('\n'))
        len3_more = 0 if "All recognized" in str(task3_more_answer) else len(task3_more_answer.split('\n'))

        len2_more = 0
        if "All recognized" not in str(task2_more_answer):
            more_2 = task2_more_answer.split('\n')
            entities = []
            for item in more_2:
                entities.extend(self._extract_entities_from_text(item))
            len2_more = len(entities)

        return [len1_more, len2_more, len3_more]

    def _get_incorrect(self, text: str, entity_relations: Any, event_entity_relations: Any,
                       event_relations: Any) -> List[str]:
        """
        Get incorrect relations/entities from the model.

        Args:
            text (str): Original text.
            entity_relations (Any): Entity relation dictionary.
            event_entity_relations (Any): Event-entity relation dictionary.
            event_relations (Any): Event relation dictionary.

        Returns:
            List[str]: [entity_rel_incorrect, event_entity_incorrect, event_rel_incorrect]
        """
        text_prompt = (
            "I need you to help me evaluate the results from information extraction, "
            "The text paragraphs are provided as followed:"
        )
        prompts = [
            "\nEvaluate the entity relationships with an emphasis on general interpretations and broader "
            "meanings rather than strict details. Accept general phrases, reasonable expressions, and contextual "
            "representations as correct, even if they are not strictly word-for-word accurate. Focus on capturing "
            "the essence of the relationships instead of precise details. If any relationships can be reasonably "
            "interpreted as correct, consider them so. All relationships should be accepted as correct unless they "
            "are explicitly unsupported by the text. If it is all correct, output \"all correct!\" only. If is "
            "not all correct, output the incorrect triples strictly line by line as the original format like"
            "{'Head': '', 'Relation': '', 'Tail': ''}.Output triples as least as possible.No blank lines.Line "
            "breaks are not allowed in a triple. The model entity relationship recognition results as follows: ",

            "\nEvaluate the following extracted entities based on the events described. Assume that every entity "
            "mentioned has a significant relevance to the events, either directly or indirectly. Recognize that "
            "general involvement, titles, and roles are valid entities, even if they appear redundant or lack "
            "explicit support. Approach the evaluation with a broad interpretation of connections and consider "
            "that all entities are relevant to the historical context. General or repeated entities should be "
            "considered correct. Similar or reasonable entities should be considered correct. Entities not central "
            "to specific events should be considered correct. Entities lack of explicit support or direct support "
            "should be considered correct. Titles and roles should be considered as correct entities, even with "
            "name redundancy. All entities should be accepted as correct unless they are explicitly unsupported "
            "by the event. If it is all correct, output \"all correct!\" only. If it is not all correct, output "
            "the incorrect triples strictly line by line as the original format like{'Event': 'sentence', "
            "'Entity': ['entity1','entity2', '...']}.Line breaks are not allowed in a triple.No blank lines."
            "Output triples as least as possible. Correct entity should be excluded from the entity list. The "
            "model named entity recognition results as followed: ",

            "\nEvaluate the relationships between the events derived from the provided text. The relationships "
            "types are :before, after, at the same time, because, and as a result. Focus on capturing the essence "
            "of the relationships instead of precise details. Consider them all correct if they logically align "
            "with the information provided, even if they are not exact. If any relationships can be reasonably "
            "interpreted or inferred as correct, consider them so. All relationships should be accepted as correct "
            "unless they are explicitly unsupported by the text. If it is all correct, output \"all correct!\" "
            "only. If is not all correct, output the incorrect triples strictly line by line as the original "
            "format like{'Head': '', 'Relation': '', 'Tail': ''}.Line breaks are not allowed in a triple.No "
            "blank lines.Output triples as least as possible. Similar or reasonable expression should be "
            "considered correct.Don't need to be same with original text word by word. The model event "
            "relationship recognition results as followed:"
        ]
        return [
            self.llm.chat(f"{text_prompt}{text}{prompts[0]}{entity_relations}", llm_config=self.llm_config),
            self.llm.chat(f"{text_prompt}{text}{prompts[1]}{event_entity_relations}", llm_config=self.llm_config),
            self.llm.chat(f"{text_prompt}{text}{prompts[2]}{event_relations}", llm_config=self.llm_config)
        ]

    def _count_incorrect(self, task1_incorrect_answer: str, task2_incorrect_answer: str,
                         task3_incorrect_answer: str) -> List[int]:
        """
        Count the number of incorrect relations/entities.

        Args:
            task1_incorrect_answer (str): Entity relation incorrect answer.
            task2_incorrect_answer (str): Event-entity relation incorrect answer.
            task3_incorrect_answer (str): Event relation incorrect answer.

        Returns:
            List[int]: [entity_rel_incorrect_count, event_entity_incorrect_count, event_rel_incorrect_count]
        """
        task1_incorrect_answer = self._remove_empty_lines(task1_incorrect_answer)
        task2_incorrect_answer = self._remove_empty_lines(task2_incorrect_answer)
        task3_incorrect_answer = self._remove_empty_lines(task3_incorrect_answer)

        len1_incorrect = 0 if "all correct" in str(task1_incorrect_answer) else len(task1_incorrect_answer.split('\n'))
        len3_incorrect = 0 if "all correct" in str(task3_incorrect_answer) else len(task3_incorrect_answer.split('\n'))

        len2_incorrect = 0
        if "all correct" not in str(task2_incorrect_answer):
            incorrect_2 = task2_incorrect_answer.split('\n')
            entities = []
            for item in incorrect_2:
                entities.extend(self._extract_entities_from_text(item))
            len2_incorrect = len(entities)

        return [len1_incorrect, len2_incorrect, len3_incorrect]

    @validate_params(
        relations=dict(validator=lambda x: _check_relations(x),
                       message="Relations cannot be empty or too many."))
    def evaluate(self, relations: List[dict]) -> None:
        """
        Run the evaluation process, compute metrics, and log results.

        Args:
            relations (List[dict]): List of relation extraction results, each containing:
                - 'raw_text'
                - 'entity_relations'
                - 'event_entity_relations'
                - 'event_relations'
        """
        precision_list_1 = []
        precision_list_2 = []
        precision_list_3 = []
        recall_list_1 = []
        recall_list_2 = []
        recall_list_3 = []
        f1_list_1 = []
        f1_list_2 = []
        f1_list_3 = []

        def process_item(args):
            i, item = args
            if isinstance(item, list):
                item = item[0]

            text = item['raw_text']
            entity_relations = item.get('entity_relations', [])
            event_entity_relations = item.get('event_entity_relations', [])
            event_relations = item.get('event_relations', [])

            len_1, len_2, len_3 = self._count_origin(entity_relations, event_entity_relations, event_relations)
            task1_more_answer, task2_more_answer, task3_more_answer = self._get_more(
                text, entity_relations, event_entity_relations, event_relations)
            len1_more, len2_more, len3_more = self._count_more(task1_more_answer, task2_more_answer, task3_more_answer)
            task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer = self._get_incorrect(
                text, entity_relations, event_entity_relations, event_relations)
            len1_incorrect, len2_incorrect, len3_incorrect = self._count_incorrect(
                task1_incorrect_answer, task2_incorrect_answer, task3_incorrect_answer)

            return ProcessResult(
                len_1=len_1, len_2=len_2, len_3=len_3,
                len1_incorrect=len1_incorrect, len2_incorrect=len2_incorrect, len3_incorrect=len3_incorrect,
                len1_more=len1_more, len2_more=len2_more, len3_more=len3_more
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for result in tqdm(
                    executor.map(process_item, enumerate(relations)),
                    total=len(relations), desc="Evaluating relations"
            ):
                results.append(result)

        for result in results:
            precision_1, recall_1, f1_1 = self._calculate(result.len_1, result.len1_incorrect, result.len1_more)
            precision_2, recall_2, f1_2 = self._calculate(result.len_2, result.len2_incorrect, result.len2_more)
            precision_3, recall_3, f1_3 = self._calculate(result.len_3, result.len3_incorrect, result.len3_more)

            precision_list_1.append(precision_1)
            precision_list_2.append(precision_2)
            precision_list_3.append(precision_3)
            recall_list_1.append(recall_1)
            recall_list_2.append(recall_2)
            recall_list_3.append(recall_3)
            f1_list_1.append(f1_1)
            f1_list_2.append(f1_2)
            f1_list_3.append(f1_3)

        average_precision_1 = np.nanmean(precision_list_1)
        average_recall_1 = np.nanmean(recall_list_1)
        average_f1_1 = np.nanmean(f1_list_1)

        average_precision_2 = np.nanmean(precision_list_2)
        average_recall_2 = np.nanmean(recall_list_2)
        average_f1_2 = np.nanmean(f1_list_2)

        average_precision_3 = np.nanmean(precision_list_3)
        average_recall_3 = np.nanmean(recall_list_3)
        average_f1_3 = np.nanmean(f1_list_3)

        logger.info(f"[average_precision_1, average_recall_1, average_f1_1]: "
                    f"{average_precision_1}, {average_recall_1}, {average_f1_1}")
        logger.info(f"[average_precision_2, average_recall_2, average_f1_2]: "
                    f"{average_precision_2}, {average_recall_2}, {average_f1_2}")
        logger.info(f"[average_precision_3, average_recall_3, average_f1_3]: "
                    f"{average_precision_3}, {average_recall_3}, {average_f1_3}")
