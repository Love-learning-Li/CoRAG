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

import unittest
from unittest.mock import Mock, patch
from mx_rag.utils.common import Lang
from langchain_core.documents import Document


from mx_rag.graphrag.relation_extraction import (
    _parse_and_repair_json,
    generate_relations_cn,
    generate_relations_en,
    LLMRelationExtractor
)


class TestParseAndRepairJson(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.chat.return_value = '[{"entity": "test", "relation": "example"}]'

    @patch('mx_rag.graphrag.relation_extraction.extract_json_like_substring')
    @patch('mx_rag.graphrag.relation_extraction.normalize_json_string')
    def test_parse_valid_json(self, mock_normalize, mock_extract):
        mock_extract.return_value = '[{"key": "value"}]'
        mock_normalize.return_value = '[{"key": "value"}]'

        result = _parse_and_repair_json(self.mock_llm, "text", "token")

        self.assertEqual(result, [{"key": "value"}])
        self.mock_llm.chat.assert_not_called()

    @patch('mx_rag.graphrag.relation_extraction.extract_json_like_substring')
    @patch('mx_rag.graphrag.relation_extraction.normalize_json_string')
    @patch('mx_rag.graphrag.relation_extraction.repair_json')
    def test_parse_with_repair_function(self, mock_repair_json, mock_normalize, mock_extract):
        mock_extract.return_value = '{"key": "value"'  # Invalid JSON
        mock_normalize.side_effect = lambda x, *args: x
        mock_repair_json.return_value = '[{"key": "value"}]'

        def repair_func(text):
            return '[{"key": "value"}]'

        result = _parse_and_repair_json(self.mock_llm, "text", "token", repair_func)

        self.assertEqual(result, [{"key": "value"}])

    @patch('mx_rag.graphrag.relation_extraction.extract_json_like_substring')
    @patch('mx_rag.graphrag.relation_extraction.normalize_json_string')
    def test_parse_with_llm_repair(self, mock_normalize, mock_extract):
        mock_extract.return_value = '{"invalid": json}'
        mock_normalize.side_effect = lambda x, *args: x
        self.mock_llm.chat.return_value = '{"invalid": "json"}'

        result = _parse_and_repair_json(self.mock_llm, "text")

        self.assertEqual(result, {"invalid": "json"})
        self.mock_llm.chat.assert_called_once()

    @patch('mx_rag.graphrag.relation_extraction.extract_json_like_substring')
    @patch('mx_rag.graphrag.relation_extraction.normalize_json_string')
    def test_parse_all_repairs_fail(self, mock_normalize, mock_extract):
        mock_extract.return_value = 'completely invalid'
        mock_normalize.side_effect = lambda x, *args: x
        self.mock_llm.chat.return_value = 'still invalid'

        result = _parse_and_repair_json(self.mock_llm, "text")

        self.assertEqual(result, [])


class TestGenerateRelations(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()

    @patch('mx_rag.graphrag.relation_extraction._parse_and_repair_json')
    def test_generate_relations_cn(self, mock_parse):
        mock_parse.return_value = [{"entity": "test"}]
        repair_func = Mock()

        result = generate_relations_cn(
            self.mock_llm, "<pad>", ["text<pad>1", "text<pad>2"], repair_func
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(mock_parse.call_count, 2)
        mock_parse.assert_any_call(self.mock_llm, "text1", "", repair_func, True, True)
        mock_parse.assert_any_call(self.mock_llm, "text2", "", repair_func, True, True)

    @patch('mx_rag.graphrag.relation_extraction._parse_and_repair_json')
    @patch('mx_rag.graphrag.relation_extraction.repair_json')
    def test_generate_relations_en(self, mock_repair_json, mock_parse):
        mock_parse.return_value = [{"entity": "test"}]

        result = generate_relations_en(self.mock_llm, ["text1", "text2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(mock_parse.call_count, 2)


class TestLLMRelationExtractor(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.mock_llm.model_name = "test_model"

    @patch('mx_rag.graphrag.relation_extraction.TRIPLE_INSTRUCTIONS_CN', {
        "entity_relation": "extract entities",
        "event_entity": "extract events",
        "event_relation": "extract event relations"
    })
    def test_init_chinese(self):
        extractor = LLMRelationExtractor(self.mock_llm, "<pad>", Lang.CH, max_workers=4)

        self.assertEqual(extractor.llm, self.mock_llm)
        self.assertEqual(extractor.pad_token, "<pad>")
        self.assertEqual(extractor.language, Lang.CH)
        self.assertEqual(extractor.max_workers, 4)
        self.assertIn("entity_relation", extractor.user_prompts)
        self.assertIn("event_entity", extractor.user_prompts)
        self.assertIn("event_relation", extractor.user_prompts)

    @patch('mx_rag.graphrag.relation_extraction.TRIPLE_INSTRUCTIONS_EN', {
        "entity_relation": "extract entities en",
        "event_entity": "extract events en",
        "event_relation": "extract event relations en"
    })
    def test_init_english_default_template(self):
        extractor = LLMRelationExtractor(self.mock_llm, "<pad>", Lang.EN)
        self.assertEqual(extractor.language, Lang.EN)

    @patch('mx_rag.graphrag.relation_extraction.generate_relations_cn')
    def test_process_relations_chinese(self, mock_generate_cn):
        mock_generate_cn.return_value = [{"test": "result"}]
        extractor = LLMRelationExtractor(self.mock_llm, "<pad>", Lang.CH)
        repair_func = Mock()

        result = extractor._process_relations(["output1", "output2"], repair_func)

        mock_generate_cn.assert_called_once_with(
            self.mock_llm, "<pad>", ["output1", "output2"], repair_func
        )
        self.assertEqual(result, [{"test": "result"}])

    @patch('mx_rag.graphrag.relation_extraction.generate_relations_en')
    def test_process_relations_english(self, mock_generate_en):
        mock_generate_en.return_value = [{"test": "result"}]
        extractor = LLMRelationExtractor(self.mock_llm, "<pad>", Lang.EN)
        repair_func = Mock()

        result = extractor._process_relations(["output1", "output2"], repair_func)

        mock_generate_en.assert_called_once_with(
            self.mock_llm, ["output1", "output2"]
        )
        self.assertEqual(result, [{"test": "result"}])

    @patch('mx_rag.graphrag.relation_extraction.fix_entity_relation_json_string')
    @patch('mx_rag.graphrag.relation_extraction.fix_entity_event_json_string')
    @patch('mx_rag.graphrag.relation_extraction.fix_event_relation_json_string')
    def test_query(self, mock_fix_event_rel, mock_fix_entity_event, mock_fix_entity_rel):
        # Setup mocks
        extractor = LLMRelationExtractor(self.mock_llm, "<pad>", max_workers=1)
        extractor._process_relations = Mock()
        extractor._process_relations.side_effect = [
            [{"entity": "rel1"}, {"entity": "rel2"}],  # entity_relations
            [{"event": "ent1"}, {"event": "ent2"}],   # event_entity_relations
            [{"event": "rel1"}, {"event": "rel2"}]    # event_relations
        ]

        # Create test documents
        docs = [
            Document(page_content="text1", metadata={"source": "file1"}, document_name="file1"),
            Document(page_content="text2", metadata={"source": "file2"}, document_name="file2")
        ]

        result = extractor.query(docs)

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["raw_text"], "text1")
        self.assertEqual(result[0]["file_id"], "file1")
        self.assertEqual(result[0]["entity_relations"], {"entity": "rel1"})
        self.assertEqual(result[0]["event_entity_relations"], {"event": "ent1"})
        self.assertEqual(result[0]["event_relations"], {"event": "rel1"})

        self.assertEqual(result[1]["raw_text"], "text2")
        self.assertEqual(result[1]["file_id"], "file2")
        self.assertEqual(result[1]["entity_relations"], {"entity": "rel2"})
        self.assertEqual(result[1]["event_entity_relations"], {"event": "ent2"})
        self.assertEqual(result[1]["event_relations"], {"event": "rel2"})

        # Verify function calls
        self.assertEqual(extractor._process_relations.call_count, 3)  # 3 relation types

if __name__ == "__main__":
    unittest.main()