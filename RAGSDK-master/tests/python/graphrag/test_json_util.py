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
import unittest

from mx_rag.graphrag.utils.json_util import (
    extract_json_like_substring,
    fix_entity_event_json_string,
    fix_entity_relation_json_string,
    fix_event_relation_json_string,
    normalize_json_string,
)


class TestFixEventRelationJsonString(unittest.TestCase):
    def test_standard_input(self):
        input_str = (
            '"头事件": "事件A", "关系": "导致", "尾事件": "事件B", '
            '"头事件": "事件C", "关系": "影响", "尾事件": "事件D"'
        )
        expected = [
            {"头事件": "事件A", "关系": "导致", "尾事件": "事件B"},
            {"头事件": "事件C", "关系": "影响", "尾事件": "事件D"}
        ]
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_single_record(self):
        input_str = '"头事件": "A", "关系": "关联", "尾事件": "B"'
        expected = [{"头事件": "A", "关系": "关联", "尾事件": "B"}]
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_with_single_quotes(self):
        input_str = "'头事件': 'X', '关系': '触发', '尾事件': 'Y'"
        # The regex expects double quotes for keys, so this should return empty
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_mixed_quotes(self):
        input_str = '"头事件": "A", "关系": \'导致\', "尾事件": "B"'
        expected = [{"头事件": "A", "关系": "导致", "尾事件": "B"}]
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_extra_spaces_and_newlines(self):
        input_str = '''
            "头事件":   "E1" ,
            "关系": "R1",
            "尾事件": "E2"
        '''
        expected = [{"头事件": "E1", "关系": "R1", "尾事件": "E2"}]
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_empty_input(self):
        input_str = ""
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_incomplete_record(self):
        input_str = '"头事件": "A", "关系": "B"'
        # No "尾事件", so no record should be appended
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_multiple_records_with_noise(self):
        input_str = (
            'Some text "头事件": "A", "关系": "B", "尾事件": "C", '
            'random "头事件": "D", "关系": "E", "尾事件": "F"'
        )
        expected = [
            {"头事件": "A", "关系": "B", "尾事件": "C"},
            {"头事件": "D", "关系": "E", "尾事件": "F"}
        ]
        result = fix_event_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)


class TestFixEntityRelationJsonString(unittest.TestCase):
    def test_standard_input(self):
        input_str = (
            '"头实体": "实体A", "关系": "属于", "尾实体": "实体B", '
            '"头实体": "实体C", "关系": "包含", "尾实体": "实体D"'
        )
        expected = [
            {"头实体": "实体A", "关系": "属于", "尾实体": "实体B"},
            {"头实体": "实体C", "关系": "包含", "尾实体": "实体D"}
        ]
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_single_record(self):
        input_str = '"头实体": "A", "关系": "关联", "尾实体": "B"'
        expected = [{"头实体": "A", "关系": "关联", "尾实体": "B"}]
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_with_single_quotes(self):
        input_str = "'头实体': 'X', '关系': '触发', '尾实体': 'Y'"
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_mixed_quotes(self):
        input_str = '"头实体": "A", "关系": \'属于\', "尾实体": "B"'
        expected = [{"头实体": "A", "关系": "属于", "尾实体": "B"}]
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_extra_spaces_and_newlines(self):
        input_str = '''
            "头实体":   "E1" ,
            "关系": "R1",
            "尾实体": "E2"
        '''
        expected = [{"头实体": "E1", "关系": "R1", "尾实体": "E2"}]
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_empty_input(self):
        input_str = ""
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_incomplete_record(self):
        input_str = '"头实体": "A", "关系": "B"'
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), [])

    def test_multiple_records_with_noise(self):
        input_str = (
            'Some text "头实体": "A", "关系": "B", "尾实体": "C", '
            'random "头实体": "D", "关系": "E", "尾实体": "F"'
        )
        expected = [
            {"头实体": "A", "关系": "B", "尾实体": "C"},
            {"头实体": "D", "关系": "E", "尾实体": "F"}
        ]
        result = fix_entity_relation_json_string(input_str)
        self.assertEqual(json.loads(result), expected)


class TestFixEntityEventJsonString(unittest.TestCase):
    def test_standard_input(self):
        input_str = (
            '"事件": "事件A", "实体": ["实体1", "实体2"], '
            '"事件": "事件B", "实体": ["实体3"]'
        )
        expected = [
            {"事件": "事件A", "实体": ["实体1", "实体2"]},
            {"事件": "事件B", "实体": ["实体3"]}
        ]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_single_record(self):
        input_str = '"事件": "A", "实体": ["X", "Y"]'
        expected = [{"事件": "A", "实体": ["X", "Y"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_with_single_quotes(self):
        input_str = '"事件": \'A\', "实体": [\'X\', \'Y\']'
        expected = [{"事件": "A", "实体": ["X", "Y"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_mixed_quotes(self):
        input_str = '"事件": "A", "实体": [\'X\', "Y"]'
        expected = [{"事件": "A", "实体": ["X", "Y"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_extra_spaces_and_newlines(self):
        input_str = '''
            "事件":   "E1" ,
            "实体": [ "A" , "B" ]
        '''
        expected = [{"事件": "E1", "实体": ["A", "B"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_empty_input(self):
        input_str = ""
        expected = []
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_incomplete_record(self):
        input_str = '"事件": "A"'
        expected = []
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_multiple_records_with_noise(self):
        input_str = (
            'Some text "事件": "A", "实体": ["B", "C"], '
            'random "事件": "D", "实体": ["E"]'
        )
        expected = [
            {"事件": "A", "实体": ["B", "C"]},
            {"事件": "D", "实体": ["E"]}
        ]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_entities_with_spaces_and_empty(self):
        input_str = '"事件": "A", "实体": ["X", " ", "", "Y"]'
        expected = [{"事件": "A", "实体": ["X", "Y"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)

    def test_entities_with_no_quotes(self):
        input_str = '"事件": "A", "实体": [X, Y]'
        expected = [{"事件": "A", "实体": ["X", "Y"]}]
        result = fix_entity_event_json_string(input_str)
        self.assertEqual(json.loads(result), expected)


class TestExtractJsonLikeSubstring(unittest.TestCase):
    def test_basic_json_array_extraction(self):
        text = "Some intro text. DATA: [1, 2, 3, 4]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[1, 2, 3, 4]")

    def test_multiple_markers_uses_last(self):
        text = "DATA: [1] ... DATA: [2, 3, 4]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[2, 3, 4]")

    def test_no_marker_returns_empty(self):
        text = "No marker here [1, 2, 3]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "")

    def test_no_brackets_returns_rest(self):
        text = "DATA: some random text without brackets"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, " some random text without brackets")

    def test_only_opening_bracket_returns_rest(self):
        text = "DATA: [incomplete array"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, " [incomplete array")

    def test_only_closing_bracket_returns_rest(self):
        text = "DATA: incomplete array]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, " incomplete array]")

    def test_nested_brackets(self):
        text = "DATA: [1, [2, 3], 4]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[1, [2, 3], 4]")

    def test_marker_at_end_returns_empty(self):
        text = "Some text DATA:"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "")

    def test_marker_with_whitespace(self):
        text = "Some text\nDATA:   [10, 20]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[10, 20]")

    def test_marker_with_multiple_arrays(self):
        text = "DATA: [1, 2] and DATA: [3, 4, 5]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[3, 4, 5]")

    def test_marker_and_brackets_with_noise(self):
        text = "Noise DATA: random [not json] more noise [1, 2, 3]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[not json] more noise [1, 2, 3]")

    def test_marker_and_brackets_with_newlines(self):
        text = "DATA:\n[\n1,\n2,\n3\n]"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "[\n1,\n2,\n3\n]")

    def test_marker_with_no_content_after(self):
        text = "DATA:"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, "")

    def test_marker_with_content_but_no_brackets(self):
        text = "DATA: just some text"
        result = extract_json_like_substring(text, "DATA:")
        self.assertEqual(result, " just some text")


class TestNormalizeJsonString(unittest.TestCase):
    def test_strip_whitespace(self):
        input_str = "   { \"a\": 1 }   "
        expected = '{ "a": 1 }'
        result = normalize_json_string(input_str)
        self.assertEqual(result, expected)

    def test_remove_newlines_tabs(self):
        input_str = '{\n"a": 1,\t"b": 2\r}'
        expected = '{"a": 1,"b": 2}'
        result = normalize_json_string(input_str)
        self.assertEqual(result, expected)

    def test_remove_all_whitespace(self):
        input_str = '  { "a" : 1 , "b" : 2 }  '
        expected = '{"a":1,"b":2}'
        result = normalize_json_string(input_str, remove_space=True)
        self.assertEqual(result, expected)

    def test_handle_single_quote(self):
        input_str = "{'a': 1, 'b': 2}"
        expected = '{"a": 1, "b": 2}'
        result = normalize_json_string(input_str, handle_single_quote=True)
        self.assertEqual(result, expected)

    def test_remove_space_and_handle_single_quote(self):
        input_str = " { 'a' : 1 , 'b' : 2 } "
        expected = '{"a":1,"b":2}'
        result = normalize_json_string(input_str, remove_space=True, handle_single_quote=True)
        self.assertEqual(result, expected)

    def test_empty_string(self):
        input_str = "   "
        expected = ""
        result = normalize_json_string(input_str)
        self.assertEqual(result, expected)

    def test_no_changes_needed(self):
        input_str = '{"a":1,"b":2}'
        expected = '{"a":1,"b":2}'
        result = normalize_json_string(input_str)
        self.assertEqual(result, expected)

    def test_only_newlines_and_tabs(self):
        input_str = "\n\t\r"
        expected = ""
        result = normalize_json_string(input_str)
        self.assertEqual(result, expected)

    def test_mixed_quotes_and_spaces(self):
        input_str = "  { 'a' : 1, \"b\" : 2 }  "
        expected = '{"a":1,"b":2}'
        result = normalize_json_string(input_str, remove_space=True, handle_single_quote=True)
        self.assertEqual(result, expected)
