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
import json


def fix_event_relation_json_string(json_str: str) -> str:
    """
    Repairs a non-standard JSON string containing event relation information
    into a valid JSON array of event relation objects.

    Each event relation object is expected to have "头事件" (head event),
    "关系" (relation), and "尾事件" (tail event) fields. 

    Args:
        json_str: The input string containing event relation information,
                  potentially in a non-standard JSON format.

    Returns:
        A JSON formatted string representing an array of event relation objects.
    """
    records = []
    pattern = r'"(头事件|关系|尾事件)":\s*("[^"]*"|\'[^\']*\')'
    matches = re.findall(pattern, json_str)

    current_record = {}
    for key, value in matches:
        current_record[key] = value[1:-1]
        if key == "尾事件":
            records.append(current_record)
            current_record = {}

    return json.dumps(records, ensure_ascii=False)


def fix_entity_relation_json_string(json_str: str) -> str:
    """
    Repairs a non-standard JSON string containing entity relation information
    into a valid JSON array of entity relation objects.

    Each entity relation object is expected to have "头实体" (head entity),
    "关系" (relation), and "尾实体" (tail entity) fields. 

    Args:
        json_str: The input string containing entity relation information,
                  potentially in a non-standard JSON format.

    Returns:
        A JSON formatted string representing an array of entity relation objects.
    """
    records = []
    pattern = r'"(头实体|关系|尾实体)":\s*("[^"]*"|\'[^\']*\')'
    matches = re.findall(pattern, json_str)

    current_record = {}
    for key, value in matches:
        current_record[key] = value[1:-1]
        if key == "尾实体":
            records.append(current_record)
            current_record = {}

    return json.dumps(records, ensure_ascii=False)


def fix_entity_event_json_string(json_str: str) -> str:
    """
    Repairs a non-standard JSON string containing entity and event information
    into a valid JSON array of objects.

    Each object is expected to have an "事件" (event) field with a string value
    and an "实体" (entity) field with a list of strings as its value. 

    Args:
        json_str: The input string containing entity and event information,
                  potentially in a non-standard JSON format.

    Returns:
        A JSON formatted string representing an array of objects, each with
        "事件" and "实体" keys.
    """
    records = []
    event_pattern = r'"事件":\s*("[^"]*"|\'[^\']*\')'
    entity_pattern = r'"实体":\s*\[([^,\[\]]*(?:,[^,\[\]]*)*)\]'

    event_matches = re.findall(event_pattern, json_str)
    entity_matches = re.findall(entity_pattern, json_str)

    # 假设事件和实体的出现顺序和数量一致
    for event_match, entity_match in zip(event_matches, entity_matches):
        event_value = event_match[1:-1]  # 去除事件值两侧的引号
        entity_list_str = entity_match.strip()
        entities = [entity.strip().strip('"').strip("'") for entity in entity_list_str.split(',')]
        entities = [e for e in entities if e.strip()]  # Remove empty strings

        records.append({"事件": event_value, "实体": entities})

    return json.dumps(records, ensure_ascii=False)


def extract_json_like_substring(text: str, start_marker: str) -> str:
    """
    Extracts a JSON-like substring from the given text.

    This function finds the last occurrence of the `start_marker` and starts
    searching from the position immediately after it. It then attempts to
    locate the content between the first '[' and the last ']'.

    If a valid '[' and ']' pair is not found, the function returns the
    substring starting from after the `start_marker` to the end of the text.

    Args:
        text: The input string to search within.
        start_marker: The marker string indicating the position from where
            the search should begin.

    Returns:
        The extracted JSON-like substring. If no valid JSON structure is
        found after the `start_marker`, it returns the portion of the text
        from after the `start_marker` to the end. Returns an empty string
        if the `start_marker` is not found in the text.
    """
    start_index = text.rfind(start_marker) if start_marker else 0
    if start_index == -1:
        return ""  # Return empty string if start marker is not found

    search_from_index = start_index + len(start_marker)
    substring = text[search_from_index:]

    json_start_index = substring.find("[")
    json_end_index = substring.rfind("]")

    if json_start_index != -1 and json_end_index > json_start_index:
        return substring[json_start_index:json_end_index + 1]
    else:
        return substring
    

def normalize_json_string(json_text: str, remove_space: bool = False, handle_single_quote: bool = False) -> str:
    """
    Normalizes a string for JSON parsing.

    Args:
        json_text: The input string to normalize.
        remove_space: If True, removes all whitespace including spaces.
        handle_single_quote: If True, replaces single quotes with double quotes.

    Returns:
        A normalized string suitable for JSON parsing.
    """
    cleaned = json_text.strip()
    if remove_space:
        cleaned = re.sub(r"\s+", "", cleaned)
    else:
        cleaned = re.sub(r"[\n\r\t]", "", cleaned)
    if handle_single_quote:
        cleaned = cleaned.replace("'", '"')
    return cleaned