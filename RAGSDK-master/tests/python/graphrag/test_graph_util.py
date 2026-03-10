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

from langchain_opengauss import openGaussAGEGraph
import pytest
import unittest
from unittest.mock import patch, MagicMock

from mx_rag.graphrag.graphs.graph_util import OpenGaussAGEAdapter, CypherQueryBuilder, cypher_value, escape_identifier


class TestOpenGaussAGEAdapter(unittest.TestCase):
    def setUp(self):
        # Patch only the __init__ of openGaussAGEGraph and OpenGaussSettings
        patcher_graph_init = patch(
            'mx_rag.graphrag.graphs.graph_util.openGaussAGEGraph.__init__', return_value=None)
        patcher_settings = patch(
            'mx_rag.graphrag.graphs.graph_util.OpenGaussSettings', autospec=True)
        self.mock_graph_init = patcher_graph_init.start()
        self.mock_settings = patcher_settings.start()
        self.addCleanup(patcher_graph_init.stop)
        self.addCleanup(patcher_settings.stop)
        self.conf = self.mock_settings()
        self.conf.host = "localhost"
        self.conf.port = 5432
        self.conf.user = "user"
        self.conf.password = "pass"
        self.conf.database = "db"
        self.age_graph = openGaussAGEGraph('test_graph', self.conf)
        self.age_graph.graph_name = 'test_graph'
        self.adapter = OpenGaussAGEAdapter(self.age_graph)

    def test_context_manager(self):
        # __enter__ returns self, __exit__ calls close
        with patch.object(self.adapter, 'close') as mock_close:
            result = self.adapter.__enter__()
            self.assertIs(result, self.adapter)
            self.adapter.__exit__(None, None, None)
            mock_close.assert_called_once()

    def test_get_cursor_yields_cursor(self):
        # _get_cursor should be called and yield its cursor
        mock_cursor = MagicMock()
        self.adapter.age_graph.connection = mock_cursor
        cursor_mock = MagicMock()
        self.adapter.age_graph.connection.cursor.return_value = cursor_mock
        with self.adapter.get_cursor() as cursor:
            self.assertIs(cursor, cursor_mock)

    def test_execute_cypher_query(self):
        # Should call self.query with the cypher query
        self.adapter.age_graph.query = MagicMock(return_value='cypher_result')
        result = self.adapter.execute_cypher_query("MATCH (n) RETURN n")
        self.adapter.age_graph.query.assert_called_once_with("MATCH (n) RETURN n")
        self.assertEqual(result, 'cypher_result')

    def test_close_closes_connection(self):
        # Should close connection if present
        mock_conn = MagicMock()
        self.adapter.age_graph.connection = mock_conn
        self.adapter.close()
        mock_conn.close.assert_called_once()

    def test_close_no_connection(self):
        # Should not fail if no connection
        self.adapter.age_graph.connection = None
        try:
            self.adapter.close()
        except Exception as e:
            self.fail(f"close() raised Exception unexpectedly: {e}")


class TestCypherQueryBuilder(unittest.TestCase):
    def test_merge_node(self):
        result = CypherQueryBuilder.merge_node({"id": "abc", "foo": 1})
        self.assertIn("CREATE (n:Node", result)
        self.assertIn("id: 'abc'", result)
        self.assertIn('foo: 1', result)

    def test_match_node(self):
        result = CypherQueryBuilder.match_node("label123")
        self.assertEqual(result, "MATCH (n:Node {id: 'label123'}) RETURN n LIMIT 1")

    def test_delete_node(self):
        result = CypherQueryBuilder.delete_node("label456")
        self.assertEqual(result, "MATCH (n:Node {id: 'label456'}) DETACH DELETE n")

    def test_match_node_properties(self):
        result = CypherQueryBuilder.match_node_properties("label789")
        self.assertEqual(result, "MATCH (n:Node {id: 'label789'}) RETURN properties(n) AS props")

    def test_match_node_attribute(self):
        result = CypherQueryBuilder.match_node_attribute("label", "foo")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'}) RETURN n.foo AS value")

    def test_set_node_attribute(self):
        result = CypherQueryBuilder.set_node_attribute("label", "foo", "bar")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'}) SET n.foo = 'bar'")

    def test_set_node_attribute_append(self):
        result = CypherQueryBuilder.set_node_attribute("label", "foo", "bar", append=True)
        self.assertIn(
            "WITH n, CASE WHEN coalesce(n.foo, '') = '' THEN 'bar' else n.foo + ',' + 'bar' END AS new_value",
            result
        )
        self.assertIn("SET n.foo = new_value", result)

    def test_set_node_attributes(self):
        props = '[{label: "abc", value: 1}, {label: "def", value: 2}]'
        result = CypherQueryBuilder.set_node_attributes("foo", props)
        self.assertIn("UNWIND", result)
        self.assertIn("MATCH (n:Node) WHERE n.id = item.label", result)
        self.assertIn("SET n.foo = item.value", result)

    def test_match_nodes_with_data(self):
        result = CypherQueryBuilder.match_nodes(True)
        self.assertEqual(result, "MATCH (n:Node) RETURN n.text AS label, properties(n) AS props")

    def test_match_nodes_without_data(self):
        result = CypherQueryBuilder.match_nodes(False)
        self.assertEqual(result, "MATCH (n:Node) RETURN n.text AS label")

    def test_match_nodes_by_attribute(self):
        result = CypherQueryBuilder.match_nodes_by_attribute("foo", "bar")
        self.assertEqual(result, 'MATCH (n:Node) WHERE n.foo = \'bar\' RETURN properties(n) AS props')

    def test_match_nodes_containing_attribute(self):
        result = CypherQueryBuilder.match_nodes_containing_attribute("foo", "bar")
        self.assertIn("toString(n.foo) CONTAINS 'bar'", result)
        self.assertIn("RETURN properties(n) AS props", result)

    def test_merge_edge_with_props(self):
        result = CypherQueryBuilder.merge_edge("src", "dst", {"relation": "KNOWS", "weight": 2})
        self.assertIn("MERGE (a)-[r:`'KNOWS'` {relation: 'KNOWS', weight: 2}]->(b)", result)

    def test_merge_edge_without_props(self):
        result = CypherQueryBuilder.merge_edge("src", "dst", {})
        self.assertIn("MERGE (a)-[r:`'related'` {}]->(b)", result)

    def test_delete_edge(self):
        result = CypherQueryBuilder.delete_edge("src", "dst")
        self.assertEqual(result, "MATCH (a:Node {id: 'src'})-[r]->(b:Node {id: 'dst'}) DELETE r")

    def test_match_edge(self):
        result = CypherQueryBuilder.match_edge("src", "dst")
        self.assertEqual(result, "MATCH (a:Node {id: 'src'})-[r]->(b:Node {id: 'dst'}) RETURN r LIMIT 1")

    def test_match_edges_with_data(self):
        result = CypherQueryBuilder.match_edges(True)
        self.assertIn(
            "RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id, properties(r) AS props",
            result
        )

    def test_match_edges_without_data(self):
        result = CypherQueryBuilder.match_edges(False)
        self.assertIn("RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id", result)
        self.assertNotIn("properties(r) AS props", result)

    def test_match_edge_attribute_with_key(self):
        result = CypherQueryBuilder.match_edge_attribute("src", "dst", "foo")
        self.assertIn("RETURN r.foo AS value", result)

    def test_match_edge_attribute_without_key(self):
        result = CypherQueryBuilder.match_edge_attribute("src", "dst")
        self.assertIn("RETURN properties(r) AS props", result)

    def test_set_edge_attribute(self):
        result = CypherQueryBuilder.set_edge_attribute("src", "dst", "foo", "bar")
        self.assertIn('SET r.foo = \'bar\'', result)

    def test_set_edge_attribute_append(self):
        result = CypherQueryBuilder.set_edge_attribute("src", "dst", "foo", "bar", append=True)
        self.assertIn("SET r.foo = coalesce(r.foo, []) + 'bar'", result)

    def test_match_edges_by_attribute(self):
        result = CypherQueryBuilder.match_edges_by_attribute("foo")
        self.assertIn("WHERE exists(r.foo)", result)
        self.assertIn("RETURN a.id as source, b.id as target, properties(r) AS props", result)

    def test_in_degree(self):
        result = CypherQueryBuilder.in_degree("label")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'})<-[r]-() RETURN count(r) AS deg")

    def test_out_degree(self):
        result = CypherQueryBuilder.out_degree("label")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'})-[r]->() RETURN count(r) AS deg")

    def test_neighbors(self):
        result = CypherQueryBuilder.neighbors("label")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'})--(m) RETURN m.text as label")

    def test_successors(self):
        result = CypherQueryBuilder.successors("label")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'})-->(m) RETURN m.text as label")

    def test_predecessors(self):
        result = CypherQueryBuilder.predecessors("label")
        self.assertEqual(result, "MATCH (n:Node {id: 'label'})<--(m) RETURN m.text as label")

    def test_count_nodes(self):
        result = CypherQueryBuilder.count_nodes()
        self.assertEqual(result, "MATCH (n:Node) RETURN count(n) AS cnt")

    def test_count_edges(self):
        result = CypherQueryBuilder.count_edges()
        self.assertEqual(result, "MATCH ()-[r]->() RETURN count(r) AS cnt")


class TestCypherValueEscaping:
    """Test the cypher_value function for proper escaping."""

    def test_string_with_single_quotes(self):
        """Test that single quotes are properly escaped."""
        result = cypher_value("test'value")
        assert result == "'test\\'value'"

    def test_string_with_backslashes(self):
        """Test that backslashes are properly escaped."""
        result = cypher_value("test\\value")
        assert result == "'test\\\\value'"

    def test_string_with_both_escapes(self):
        """Test that both backslashes and quotes are escaped correctly."""
        result = cypher_value("test\\'value")
        assert result == "'test\\\\\\'value'"

    def test_malicious_string_with_cypher_commands(self):
        """Test that malicious strings with Cypher commands are escaped."""
        malicious = "'}}) DETACH DELETE (n) //"
        result = cypher_value(malicious)
        # Should be escaped so it can't break out of the string literal
        assert result == "'\\'}}\\\\) DETACH DELETE \\\\(n\\\\) //'"

    def test_integer_values(self):
        """Test that integers are converted correctly."""
        assert cypher_value(42) == "42"
        assert cypher_value(-100) == "-100"

    def test_float_values(self):
        """Test that floats are converted correctly."""
        assert cypher_value(3.14) == "3.14"
        assert cypher_value(-2.5) == "-2.5"

    def test_boolean_values(self):
        """Test that booleans are converted correctly."""
        assert cypher_value(True) == "true"
        assert cypher_value(False) == "false"

    def test_none_value(self):
        """Test that None is converted to null."""
        assert cypher_value(None) == "null"

    def test_list_values(self):
        """Test that lists are properly formatted."""
        result = cypher_value([1, "test", True])
        assert result == "[1, 'test', true]"

    def test_dict_values(self):
        """Test that dictionaries are properly formatted."""
        result = cypher_value({"key": "value", "num": 42})
        assert "key: 'value'" in result
        assert "num: 42" in result


class TestEscapeIdentifier:
    """Test the escape_identifier function for identifier validation."""

    def test_valid_identifier_alphanumeric(self):
        """Test that valid alphanumeric identifiers pass."""
        assert escape_identifier("valid_identifier") == "valid_identifier"
        assert escape_identifier("test123") == "test123"

    def test_valid_identifier_with_hyphen(self):
        """Test that identifiers with hyphens are allowed."""
        assert escape_identifier("test-key") == "test-key"

    def test_identifier_with_special_chars_rejected(self):
        """Test that identifiers with special characters are rejected."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            escape_identifier("test.key")

        with pytest.raises(ValueError, match="Invalid identifier"):
            escape_identifier("test;DROP")

        with pytest.raises(ValueError, match="Invalid identifier"):
            escape_identifier("test'key")

    def test_identifier_starting_with_digit_rejected(self):
        """Test that identifiers starting with digits are rejected."""
        with pytest.raises(ValueError, match="Cannot start with a digit"):
            escape_identifier("123test")

    def test_empty_identifier_rejected(self):
        """Test that empty identifiers are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            escape_identifier("")

    def test_non_string_identifier_rejected(self):
        """Test that non-string identifiers are rejected."""
        with pytest.raises(ValueError, match="must be a string"):
            escape_identifier(123)


class TestCypherQueryBuilderInjectionProtection:
    """Test that CypherQueryBuilder methods prevent injection attacks."""

    def test_match_node_injection_protection(self):
        """Test that match_node properly escapes label parameter."""
        malicious_label = '"}}) DETACH DELETE (n) //'
        query = CypherQueryBuilder.match_node(malicious_label)

        # The malicious string should be escaped and not break the query
        assert "DETACH DELETE" in query  # The text is there
        assert "'}}" not in query  # But not as executable code
        assert query.startswith("MATCH (n:Node {id: '")

    def test_delete_node_injection_protection(self):
        """Test that delete_node properly escapes label parameter."""
        malicious_label = '"}}) MATCH (x) DETACH DELETE x //'
        query = CypherQueryBuilder.delete_node(malicious_label)

        # Verify it's properly escaped
        assert query.startswith("MATCH (n:Node {id: '")
        assert "\\'" in query or "\\)" in query  # Escaped characters

    def test_match_node_attribute_key_validation(self):
        """Test that match_node_attribute validates property keys."""
        # Valid key should work
        query = CypherQueryBuilder.match_node_attribute("test_label", "valid_key")
        assert "RETURN n.valid_key AS value" in query

        # Invalid key should raise error
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.match_node_attribute("label", "invalid;key")

    def test_set_node_attribute_injection_protection(self):
        """Test that set_node_attribute protects against injection."""
        malicious_label = '"}}) MATCH (x) SET x.evil = true //'
        malicious_key = "key; DROP"

        # Key validation should prevent malicious key
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.set_node_attribute("label", malicious_key, "value")

        # Label should be escaped
        query = CypherQueryBuilder.set_node_attribute(malicious_label, "valid_key", "value")
        assert "\\'" in query or "\\)" in query

    def test_merge_edge_relation_injection_protection(self):
        """Test that merge_edge validates relation names."""
        attributes = {
            "relation": "'}}) DETACH DELETE (a) //'",
            "weight": 1.0
        }
        query = CypherQueryBuilder.merge_edge("source", "target", attributes)
        assert "'\\'}}\\\\) DETACH DELETE \\\\(a\\\\) //\\''" in query

    def test_merge_edge_label_injection_protection(self):
        """Test that merge_edge escapes source and target labels."""
        malicious_source = '"}}) DETACH DELETE (a) //'
        malicious_target = '"}}) CREATE (evil:Backdoor) //'
        attributes = {"relation": "test"}

        query = CypherQueryBuilder.merge_edge(malicious_source, malicious_target, attributes)

        # Verify labels are escaped
        assert "\\'" in query or "\\)" in query
        assert query.count("MATCH") == 1  # Only one MATCH statement

    def test_match_edge_attribute_protection(self):
        """Test that match_edge_attribute validates keys."""
        # Valid key should work
        query = CypherQueryBuilder.match_edge_attribute("src", "tgt", "valid_key")
        assert "RETURN r.valid_key AS value" in query

        # Invalid key should raise error
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.match_edge_attribute("src", "tgt", "invalid.key")

    def test_set_edge_attribute_protection(self):
        """Test that set_edge_attribute protects against injection."""
        # Key validation
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.set_edge_attribute("src", "tgt", "bad;key", "value")

        # Valid usage with malicious value (should be escaped)
        malicious_value = "'}}) DETACH DELETE (r) //"
        query = CypherQueryBuilder.set_edge_attribute("src", "tgt", "valid_key", malicious_value)
        assert "\\'" in query or "\\)" in query

    def test_match_edges_by_attribute_protection(self):
        """Test that match_edges_by_attribute validates keys."""
        # Valid key
        query = CypherQueryBuilder.match_edges_by_attribute("valid_key")
        assert "exists(r.valid_key)" in query

        # Invalid key
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.match_edges_by_attribute("bad.key")

    def test_match_nodes_by_attribute_protection(self):
        """Test that match_nodes_by_attribute validates keys and escapes values."""
        # Valid key with potentially malicious value
        malicious_value = "'}}) DETACH DELETE (n) //"
        query = CypherQueryBuilder.match_nodes_by_attribute("valid_key", malicious_value)
        assert "WHERE n.valid_key = " in query
        assert "\\'" in query or "\\)" in query

        # Invalid key
        with pytest.raises(ValueError, match="Invalid identifier"):
            CypherQueryBuilder.match_nodes_by_attribute("bad;key", "value")

    def test_degree_methods_injection_protection(self):
        """Test that degree methods properly escape labels."""
        malicious_label = '"}}) MATCH (x) DETACH DELETE x //'

        # Test all degree methods
        in_deg_query = CypherQueryBuilder.in_degree(malicious_label)
        out_deg_query = CypherQueryBuilder.out_degree(malicious_label)
        neighbors_query = CypherQueryBuilder.neighbors(malicious_label)
        successors_query = CypherQueryBuilder.successors(malicious_label)
        predecessors_query = CypherQueryBuilder.predecessors(malicious_label)

        # All should have escaped labels
        for query in [in_deg_query, out_deg_query, neighbors_query,
                      successors_query, predecessors_query]:
            assert "\\'" in query or "\\)" in query


class TestComplexInjectionScenarios:
    """Test complex injection attack scenarios."""

    def test_nested_quotes_in_dict_values(self):
        """Test that nested quotes in dictionary values are escaped."""
        attributes = {
            "name": "test'value",
            "description": "complex'nested\"quotes"
        }
        result = cypher_value(attributes)
        # Should have proper escaping
        assert "\\'" in result

    def test_unicode_and_special_characters(self):
        """Test that unicode and special characters are handled safely."""
        unicode_string = "测试'数据"
        result = cypher_value(unicode_string)
        assert result.startswith("'")
        assert result.endswith("'")
        assert "\\'" in result

    def test_very_long_injection_attempt(self):
        """Test that very long injection attempts are still escaped."""
        long_malicious = "'" + "x" * 10000 + "'}}) DETACH DELETE (n) //"
        result = cypher_value(long_malicious)
        # Should be properly escaped
        assert result.startswith("'")
        assert "\\'" in result
