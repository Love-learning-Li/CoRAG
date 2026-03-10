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
from unittest.mock import MagicMock, call, patch
from paddle.base import libpaddle
from mx_rag.graphrag.graphs.opengauss_graph import OpenGaussGraph


class TestOpenGaussGraph(unittest.TestCase):
    def setUp(self):
        # Patch OpenGaussAGEAdapter and CypherQueryBuilder for isolation
        patcher_adapter = patch('mx_rag.graphrag.graphs.opengauss_graph.OpenGaussAGEAdapter', autospec=True)
        patcher_settings = patch('mx_rag.graphrag.graphs.opengauss_graph.openGaussAGEGraph', autospec=True)
        self.mock_adapter_cls = patcher_adapter.start()
        self.mock_settings_cls = patcher_settings.start()
        self.addCleanup(patcher_adapter.stop)
        self.addCleanup(patcher_settings.stop)
        self.mock_adapter = self.mock_adapter_cls.return_value
        self.mock_adapter.connection = MagicMock() 
        self.age_graph = self.mock_settings_cls.return_value
        self.graph = OpenGaussGraph('test_graph', self.age_graph)

    def test_add_node_calls_adapter(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.has_node = MagicMock(side_effect=[False])
        self.graph.add_node("foo", foo_attr=1)
        self.mock_adapter.execute_cypher_query.assert_called_once()
        args = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn("CREATE (n:Node", args)
        self.assertIn("foo_attr", args)

    def test_save_writes_json(self):
        self.graph.get_nodes = MagicMock(return_value=[("foo", {"id": "id1", "text": "foo"})])
        self.graph.get_edges = MagicMock(return_value=[("foo", "bar", {"start_id": "id1", "end_id": "id2"})])
        with patch("os.fdopen", unittest.mock.mock_open()) as m:
            self.graph.save("dummy.json")
            handle = m()
            written = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertIn('"nodes"', written)
            self.assertIn('"links"', written)
            self.assertIn('"foo"', written)

    def test_add_nodes_from_tuple_and_str(self):
        self.graph.has_node = MagicMock(side_effect=[False, False])
        self.graph.add_node = MagicMock()
        self.graph.add_nodes_from([("foo", {"a": 1}), "bar"], common=2)
        self.graph.add_node.assert_any_call("foo", a=1, common=2)
        self.graph.add_node.assert_any_call("bar", common=2)
        self.assertEqual(self.graph.add_node.call_count, 2)

    def test_remove_node_calls_adapter_if_exists(self):
        self.graph.has_node = MagicMock(return_value=True)
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.remove_node("foo")
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_remove_node_does_nothing_if_not_exists(self):
        self.graph.has_node = MagicMock(return_value=False)
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.remove_node("foo")
        self.mock_adapter.execute_cypher_query.assert_not_called()

    def test_has_node_true_false(self):
        self.mock_adapter.execute_cypher_query.return_value = [{}]
        self.assertTrue(self.graph.has_node("foo"))
        self.mock_adapter.execute_cypher_query.return_value = []
        self.assertFalse(self.graph.has_node("foo"))

    def test_has_node_type_error(self):
        self.mock_adapter.execute_cypher_query.side_effect = TypeError("Type Error")
        self.assertFalse(self.graph.has_node("foo"))
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_has_node_attribute_error(self):
        self.mock_adapter.execute_cypher_query.side_effect = AttributeError("Attribute Error")
        self.assertFalse(self.graph.has_node("foo"))
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_has_node_general_exception(self):
        error_message = "does not exist"
        self.mock_adapter.execute_cypher_query.side_effect = Exception(error_message)
        self.assertFalse(self.graph.has_node("foo"))
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_has_node_unexpected_exception(self):
        self.mock_adapter.execute_cypher_query.side_effect = Exception("Unexpected Error")
        with self.assertRaises(Exception):
            self.graph.has_node("foo")
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_get_node_attributes_all_and_key(self):
        self.mock_adapter.execute_cypher_query.side_effect = [
            [{"props": {"a": 1}}],
            [{"value": 2}]
        ]
        self.assertEqual(self.graph.get_node_attributes("foo"), {"a": 1})
        self.assertEqual(self.graph.get_node_attributes("foo", "bar"), 2)

    def test_set_node_attributes_empty(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.set_node_attributes({}, "foo")
        self.mock_adapter.execute_cypher_query.assert_not_called()

    def test_set_node_attributes_nonempty(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.set_node_attributes({"foo": 1, "bar": 2}, "baz")
        self.mock_adapter.execute_cypher_query.assert_called_once()
        arg = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn("UNWIND", arg)
        self.assertIn("SET n.baz = item.value", arg)

    def test_update_node_attribute_append_false(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.get_node_attributes = MagicMock()
        self.graph.update_node_attribute("foo", "bar", "baz", append=False)
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_update_node_attribute_append_true(self):
        self.graph.get_node_attributes = MagicMock(return_value="a,b")
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.update_node_attribute("foo", "bar", "c", append=True)
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_update_node_attributes_batch_no_updates(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.update_node_attributes_batch([])
        self.mock_adapter.execute_cypher_query.assert_not_called()

    def test_update_node_attributes_batch_append_false(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        updates = [("foo", {"a": "1"}), ("bar", {"a": "2"})]
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_node_attributes_batch(updates, batch_size=1, append=False)
        self.assertTrue(self.mock_adapter.execute_cypher_query.called)

    def test_update_node_attributes_batch_append_true(self):
        self.graph.get_node_attributes = MagicMock(return_value={"a": "1"})
        updates = [("foo", {"a": "2"})]
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_node_attributes_batch(updates, batch_size=1, append=True)
        self.assertTrue(self.mock_adapter.execute_cypher_query.called)

    def test_update_edge_attributes_batch_no_updates(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.update_edge_attributes_batch([])
        self.mock_adapter.execute_cypher_query.assert_not_called()

    def test_update_edge_attributes_batch_append_false(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        updates = [
            ("foo", "bar", {"a": "1"}),
            ("baz", "qux", {"a": "2", "b": "3"})
        ]
        # Patch tqdm to passthrough
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_edge_attributes_batch(updates, batch_size=1, append=False)
        self.assertTrue(self.mock_adapter.execute_cypher_query.called)
        # Should be called twice (batch_size=1, 2 updates)
        self.assertEqual(self.mock_adapter.execute_cypher_query.call_count, 2)
        # Check cypher query structure
        for call_args in self.mock_adapter.execute_cypher_query.call_args_list:
            cypher = call_args[0][0]
            self.assertIn("UNWIND", cypher)
            self.assertIn("MATCH (a:Node)-[r]->(b:Node)", cypher)
            self.assertIn("SET", cypher)

    def test_update_edge_attributes_batch_append_true_merges(self):
        # Simulate get_edge_attributes returning existing values
        self.graph.get_edge_attributes = MagicMock(side_effect=[
            {"a": "1,2"}, {"a": "2,3", "b": "x"}
        ])
        updates = [
            ("foo", "bar", {"a": "2,3"}),
            ("baz", "qux", {"a": "4", "b": "y"})
        ]
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_edge_attributes_batch(updates, batch_size=2, append=True)
        self.assertTrue(self.mock_adapter.execute_cypher_query.called)
        # Check that merged values are deduplicated and sorted
        cypher = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn('a: "1,2,3"', cypher)  # 1,2 + 2,3 => 1,2,3
        self.assertIn('a: "2,3,4"', cypher)  # 2,3 + 4 => 2,3,4
        self.assertIn('b: "x,y"', cypher)   # x + y => x,y

    def test_update_edge_attributes_batch_handles_empty_old_and_new(self):
        self.graph.get_edge_attributes = MagicMock(return_value={})
        updates = [("foo", "bar", {"a": ""})]
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_edge_attributes_batch(updates, batch_size=1, append=True)
        cypher = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn('a: ""', cypher)

    def test_update_edge_attributes_batch_multiple_keys(self):
        self.graph.get_edge_attributes = MagicMock(return_value={"a": "1", "b": "2"})
        updates = [("foo", "bar", {"a": "2", "b": "3"})]
        with patch("mx_rag.graphrag.graphs.opengauss_graph.tqdm", lambda x, **k: x):
            self.graph.update_edge_attributes_batch(updates, batch_size=1, append=True)
        cypher = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn('a: "1,2"', cypher)
        self.assertIn('b: "2,3"', cypher)

    def test_get_nodes_with_data(self):
        # Mock the adapter's execute_cypher_query method
        self.mock_adapter.execute_cypher_query.return_value = [
            {"label": "node1", "props": {"key1": "value1"}},
            {"label": "node2", "props": {"key2": "value2"}}
        ]

        # Call the method
        result = self.graph.get_nodes(with_data=True)

        # Assertions
        self.mock_adapter.execute_cypher_query.assert_called_once_with(
            "MATCH (n:Node) RETURN n.text AS label, properties(n) AS props")
        self.assertEqual(result, [("node1", {"key1": "value1"}), ("node2", {"key2": "value2"})])

    def test_get_nodes_without_data(self):
        # Mock the adapter's execute_cypher_query method
        self.mock_adapter.execute_cypher_query.return_value = [
            {"label": "node1"},
            {"label": "node2"}
        ]

        # Call the method
        result = self.graph.get_nodes(with_data=False)

        # Assertions
        self.mock_adapter.execute_cypher_query.assert_called_once_with("MATCH (n:Node) RETURN n.text AS label")
        self.assertEqual(result, ["node1", "node2"])

    def test_get_nodes_by_attribute(self):
        # Mock the adapter's execute_cypher_query method
        self.mock_adapter.execute_cypher_query.return_value = [
            {"props": {"text": "node1"}},
            {"props": {"text": "node2"}}
        ]

        # Call the method
        result = self.graph.get_nodes_by_attribute("key", "value")

        # Assertions
        self.mock_adapter.execute_cypher_query.assert_called_once_with(
            "MATCH (n:Node) WHERE n.key = 'value' RETURN properties(n) AS props")
        self.assertEqual(result, ["node1", "node2"])

    def test_get_nodes_containing_attribute_value(self):
        # Mock the adapter's execute_cypher_query method
        self.mock_adapter.execute_cypher_query.return_value = [
            {"props": {"text": "node1"}},
            {"props": {"text": "node2"}}
        ]

        # Call the method
        result = self.graph.get_nodes_containing_attribute_value("key", "substring")

        # Assertions
        self.mock_adapter.execute_cypher_query.assert_called_once_with(
            'MATCH (n:Node) WHERE toString(n.key) CONTAINS \'substring\' RETURN properties(n) AS props'
        )
        self.assertEqual(result, ["node1", "node2"])

    def test_add_edge(self):
        # Mock has_node and add_node methods
        self.graph.has_node = MagicMock(side_effect=[True, False])
        self.graph.has_edge = MagicMock(side_effect=[False])
        self.graph.add_node = MagicMock()

        # Call the method
        self.graph.add_edge("node1", "node2", relation="KNOWS")

        # Assertions
        self.graph.has_node.assert_any_call("node1")
        self.graph.has_node.assert_any_call("node2")
        self.graph.add_node.assert_called_once_with("node2")
        self.mock_adapter.execute_cypher_query.assert_called_once()
        query = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn("MERGE (a)-[r:`'KNOWS'`", query)

    def test_add_edge_syntax_error(self):
        # Mock has_node and has_edge methods
        self.graph.has_node = MagicMock(side_effect=[True, True])
        self.graph.has_edge = MagicMock(side_effect=[False])
        self.mock_adapter.execute_cypher_query.side_effect = SyntaxError("Syntax Error")

        # Call the method and expect it to raise SyntaxError
        with self.assertRaises(SyntaxError):
            self.graph.add_edge("node1", "node2", relation="KNOWS")

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_add_edge_connection_error(self):
        # Mock has_node and has_edge methods
        self.graph.has_node = MagicMock(side_effect=[True, True])
        self.graph.has_edge = MagicMock(side_effect=[False])
        self.mock_adapter.execute_cypher_query.side_effect = ConnectionError("Connection Error")

        # Call the method and expect it to raise ConnectionError
        with self.assertRaises(ConnectionError):
            self.graph.add_edge("node1", "node2", relation="KNOWS")

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_add_edge_general_exception(self):
        # Mock has_node and has_edge methods
        self.graph.has_node = MagicMock(side_effect=[True, True])
        self.graph.has_edge = MagicMock(side_effect=[False])
        self.mock_adapter.execute_cypher_query.side_effect = Exception("General Error")

        # Call the method and expect it to raise Exception
        with self.assertRaises(Exception):
            self.graph.add_edge("node1", "node2", relation="KNOWS")

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_add_edges_from(self):
        # Mock add_edge method
        self.graph.add_edge = MagicMock()

        # Call the method
        edges = [("node1", "node2"), ("node3", "node4", {"relation": "KNOWS"})]
        self.graph.add_edges_from(edges)

        # Assertions
        self.graph.add_edge.assert_any_call("node1", "node2")
        self.graph.add_edge.assert_any_call("node3", "node4", relation="KNOWS")
        self.assertEqual(self.graph.add_edge.call_count, 2)

    def test_remove_edge(self):
        # Mock has_edge method
        self.graph.has_edge = MagicMock(return_value=True)

        # Call the method
        self.graph.remove_edge("node1", "node2")

        # Assertions
        self.graph.has_edge.assert_called_once_with("node1", "node2")
        self.mock_adapter.execute_cypher_query.assert_called_once()
        query = self.mock_adapter.execute_cypher_query.call_args[0][0]
        self.assertIn("DELETE r", query)

    def test_has_edge(self):
        # Mock the adapter's execute_cypher_query method
        self.mock_adapter.execute_cypher_query.return_value = [{}]

        # Call the method
        result = self.graph.has_edge("node1", "node2")

        # Assertions
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertTrue(result)

        # Test when edge does not exist
        self.mock_adapter.execute_cypher_query.return_value = []
        result = self.graph.has_edge("node1", "node2")
        self.assertFalse(result)

    def test_has_edge_syntax_error(self):
        # Mock the adapter's execute_cypher_query to raise SyntaxError
        self.mock_adapter.execute_cypher_query.side_effect = SyntaxError("Syntax Error")

        # Call the method and expect it to raise SyntaxError
        with self.assertRaises(SyntaxError):
            self.graph.has_edge("node1", "node2")

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_has_edge_value_error(self):
        # Mock the adapter's execute_cypher_query to raise ValueError
        self.mock_adapter.execute_cypher_query.side_effect = ValueError("Invalid Value")

        # Call the method and expect it to raise ValueError
        with self.assertRaises(ValueError):
            self.graph.has_edge("node1", "node2")

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_has_edge_general_exception(self):
        # Mock the adapter's execute_cypher_query to raise a general Exception
        error_message = "does not exist"
        self.mock_adapter.execute_cypher_query.side_effect = Exception(error_message)

        # Call the method and expect it to return False
        result = self.graph.has_edge("node1", "node2")
        self.assertFalse(result)

        # Verify that execute_cypher_query was called
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_get_edge_attributes_with_key(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'value': 'test_value'}]
        result = self.graph.get_edge_attributes('node1', 'node2', 'key1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, 'test_value')

    def test_get_edge_attributes_without_key(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'props': {'key1': 'value1', 'key2': 'value2'}}]
        result = self.graph.get_edge_attributes('node1', 'node2')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, {'key1': 'value1', 'key2': 'value2'})

    def test_update_edge_attribute_append_false(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.has_edge = MagicMock(return_value=True)
        self.graph.update_edge_attribute('node1', 'node2', 'key1', 'value1', append=False)
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_update_edge_attribute_append_true(self):
        self.mock_adapter.execute_cypher_query.reset_mock()
        self.graph.has_edge = MagicMock(return_value=True)
        self.graph.update_edge_attribute('node1', 'node2', 'key1', 'value1', append=True)
        self.mock_adapter.execute_cypher_query.assert_called_once()

    def test_get_edges_with_data(self):
        self.mock_adapter.execute_cypher_query.return_value = [
            {
                "source": "node1",
                "target": "node2",
                "start_id": "id1",
                "end_id": "id2",
                "props": {"key1": "value1"}
            }
        ]
        result = self.graph.get_edges(with_data=True)
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, [('node1', 'node2', {'start_id': 'id1', 'end_id': 'id2', 'key1': 'value1'})])

    def test_get_edges_without_data(self):
        self.mock_adapter.execute_cypher_query.return_value = [
            {
                "source": "node1",
                "target": "node2",
                "start_id": "id1",
                "end_id": "id2"
            }
        ]
        result = self.graph.get_edges(with_data=False)
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, [('node1', 'node2', {'start_id': 'id1', 'end_id': 'id2'})])

    def test_get_edge_attribute_values(self):
        self.mock_adapter.execute_cypher_query.return_value = [
            {'props': {'key1': 'value1'}},
            {'props': {'key1': 'value2'}}
        ]
        result = self.graph.get_edge_attribute_values('key1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, ['value1', 'value2'])

    def test_in_degree(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'deg': 3}]
        result = self.graph.in_degree('node1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, 3)

    def test_out_degree(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'deg': 2}]
        result = self.graph.out_degree('node1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, 2)

    def test_neighbors(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'label': 'node2'}, {'label': 'node3'}]
        result = self.graph.neighbors('node1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, ['node2', 'node3'])

    def test_successors(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'label': 'node2'}, {'label': 'node3'}]
        result = self.graph.successors('node1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, ['node2', 'node3'])

    def test_predecessors(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'label': 'node2'}, {'label': 'node3'}]
        result = self.graph.predecessors('node1')
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, ['node2', 'node3'])

    def test_number_of_nodes(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'cnt': 5}]
        result = self.graph.number_of_nodes()
        self.mock_adapter.execute_cypher_query.assert_called_once_with("MATCH (n:Node) RETURN count(n) AS cnt")
        self.assertEqual(result, 5)

    def test_number_of_edges(self):
        self.mock_adapter.execute_cypher_query.return_value = [{'cnt': 10}]
        result = self.graph.number_of_edges()
        self.mock_adapter.execute_cypher_query.assert_called_once_with("MATCH ()-[r]->() RETURN count(r) AS cnt")
        self.assertEqual(result, 10)

    def test_density(self):
        self.graph.number_of_nodes = MagicMock(return_value=5)
        self.graph.number_of_edges = MagicMock(return_value=10)
        result = self.graph.density()
        self.assertEqual(result, 0.5)

    def test_density_with_one_node(self):
        self.graph.number_of_nodes = MagicMock(return_value=1)
        self.graph.number_of_edges = MagicMock(return_value=0)
        result = self.graph.density()
        self.assertEqual(result, 0.0)

    def test_connected_components_weakly(self):
        self.graph._find_weakly_connected_components = MagicMock(return_value=[{'A', 'B'}, {'C'}])
        result = self.graph.connected_components(is_directed=True)
        self.graph._find_weakly_connected_components.assert_called_once()
        self.assertEqual(result, [{'A', 'B'}, {'C'}])

    def test_connected_components_strongly(self):
        self.graph._find_strongly_connected_components = MagicMock(return_value=[{'A'}, {'B', 'C'}])
        result = self.graph.connected_components(is_directed=False)
        self.graph._find_strongly_connected_components.assert_called_once()
        self.assertEqual(result, [{'A'}, {'B', 'C'}])

    def test_create_index_for_edge_no_relations(self):
        self.graph.get_edge_attribute_values = MagicMock(return_value=[])
        self.graph.create_index_for_edge()
        self.graph.get_edge_attribute_values.assert_called_once_with("relation")
        self.mock_adapter.execute_cypher_query.assert_not_called()

    def test_create_index_for_edge_with_relations(self):
        self.graph.get_edge_attribute_values = MagicMock(return_value=["relation1", "relation2"])
        self.graph._create_relation_indexes = MagicMock(return_value=True)
        self.graph.create_index_for_edge()
        self.graph._create_relation_indexes.assert_any_call("relation1", [], [])
        self.graph._create_relation_indexes.assert_any_call("relation2", [], [])
        self.assertEqual(self.graph._create_relation_indexes.call_count, 2)

    def test_create_index_for_node(self):
        self.graph._get_node_index_definitions = MagicMock(return_value=[
            ("index1", "CREATE INDEX index1 ON Node(id);", False),
            ("index2", "CREATE INDEX index2 ON Node(properties);", True)
        ])
        self.graph._create_single_node_index = MagicMock(return_value=True)
        self.graph.create_index_for_node()
        self.graph._create_single_node_index.assert_any_call("CREATE INDEX index1 ON Node(id);", False, "index1")
        self.graph._create_single_node_index.assert_any_call("CREATE INDEX index2 ON Node(properties);", True, "index2")
        self.assertEqual(self.graph._create_single_node_index.call_count, 2)

    def test_reset(self):
        mock_cursor = MagicMock()
        mock_connection = MagicMock()  # Mock the connection
        self.mock_adapter.get_cursor.return_value.__enter__.return_value = mock_cursor
        self.mock_adapter.connection = mock_connection  # Assign the mocked connection
        self.graph.reset()
        mock_cursor.execute.assert_any_call("SELECT * FROM ag_catalog.drop_graph(%s, true)", ('test_graph',))
        self.assertEqual(mock_connection.commit.call_count, 1)
        self.assertEqual(mock_cursor.execute.call_count, 1)

    def test_subgraph(self):
        self.mock_adapter.execute_cypher_query.return_value = [
            {'source': 'A', 'relation': 'KNOWS', 'target': 'B'},
            {'source': 'B', 'relation': 'FRIENDS', 'target': 'C'}
        ]
        result = self.graph.subgraph(['A', 'B'], depth=2)
        self.mock_adapter.execute_cypher_query.assert_called_once()
        self.assertEqual(result, [('A', 'KNOWS', 'B'), ('B', 'FRIENDS', 'C')])

    def test_get_node_index_definitions(self):
        index_definitions = self.graph._get_node_index_definitions()
        self.assertEqual(len(index_definitions), 3)
        self.assertEqual(index_definitions[0][0], "index_graph_node_id")
        self.assertIn("CREATE INDEX IF NOT EXISTS", index_definitions[0][1])
        self.assertFalse(index_definitions[0][2])

    def test_create_single_node_index_success(self):
        self.mock_adapter.connection = MagicMock()
        self.graph._execute_regular_index = MagicMock()
        self.graph._execute_concurrent_index = MagicMock()

        result = self.graph._create_single_node_index("CREATE INDEX", False, "index_name")
        self.assertTrue(result)
        self.graph._execute_regular_index.assert_called_once_with("CREATE INDEX")

    def test_create_single_node_index_failure(self):
        self.mock_adapter.connection = MagicMock()
        self.graph._execute_regular_index = MagicMock(side_effect=Exception("Error"))
        self.graph._execute_concurrent_index = MagicMock()

        result = self.graph._create_single_node_index("CREATE INDEX", False, "index_name")
        self.assertFalse(result)

    def test_create_single_node_index_syntax_error(self):
        self.mock_adapter.connection = MagicMock()
        self.graph._execute_regular_index = MagicMock(side_effect=SyntaxError("Syntax Error"))
        self.graph._execute_concurrent_index = MagicMock()

        result = self.graph._create_single_node_index("CREATE INDEX", False, "index_name")
        self.assertFalse(result)

    def test_create_single_node_index_value_error(self):
        self.mock_adapter.connection = MagicMock()
        self.graph._execute_regular_index = MagicMock(side_effect=ValueError("Value Error"))
        self.graph._execute_concurrent_index = MagicMock()

        result = self.graph._create_single_node_index("CREATE INDEX", False, "index_name")
        self.assertFalse(result)

    def test_execute_concurrent_index(self):
        mock_conn = MagicMock()
        self.mock_adapter.connection = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        self.graph._execute_concurrent_index("CREATE INDEX")
        mock_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("CREATE INDEX")
        self.assertTrue(mock_conn.autocommit)

    def test_execute_regular_index(self):
        mock_cursor = MagicMock()
        self.mock_adapter.get_cursor.return_value.__enter__.return_value = mock_cursor

        self.graph._execute_regular_index("CREATE INDEX")
        mock_cursor.execute.assert_called_once_with("CREATE INDEX")
        self.mock_adapter.connection.commit.assert_called_once()

    def test_execute_regular_index_syntax_error(self):
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = SyntaxError("Syntax Error")
        self.mock_adapter.get_cursor.return_value.__enter__.return_value = mock_cursor

        with self.assertRaises(SyntaxError):
            self.graph._execute_regular_index("CREATE INDEX")

        mock_cursor.execute.assert_called_once_with("CREATE INDEX")
        self.mock_adapter.connection.rollback.assert_called_once()
        self.mock_adapter.connection.commit.assert_not_called()

    def test_execute_regular_index_connection_error(self):
        self.mock_adapter.get_cursor.side_effect = ConnectionError("Connection Error")

        with self.assertRaises(ConnectionError):
            self.graph._execute_regular_index("CREATE INDEX")

        self.mock_adapter.connection.rollback.assert_called_once()
        self.mock_adapter.connection.commit.assert_not_called()

    def test_execute_regular_index_general_exception(self):
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("General Error")
        self.mock_adapter.get_cursor.return_value.__enter__.return_value = mock_cursor

        with self.assertRaises(Exception):
            self.graph._execute_regular_index("CREATE INDEX")

        mock_cursor.execute.assert_called_once_with("CREATE INDEX")
        self.mock_adapter.connection.rollback.assert_called_once()
        self.mock_adapter.connection.commit.assert_not_called()


    def test_generate_safe_index_prefix(self):
        prefix = self.graph._generate_safe_index_prefix("relation-name")
        self.assertTrue(prefix.startswith("relation_name_"))
        self.assertEqual(len(prefix.split("_")[-1]), 8)

    def test_build_index_queries(self):
        queries = self.graph._build_index_queries("relation", "prefix")
        self.assertEqual(len(queries), 2)
        self.assertIn("CREATE INDEX IF NOT EXISTS index_prefix_start", queries[0])
        self.assertIn("CREATE INDEX IF NOT EXISTS index_prefix_end", queries[1])

    def test_create_relation_indexes_success(self):
        self.graph._execute_index_queries = MagicMock()
        successful_indexes = []
        failed_relations = []

        result = self.graph._create_relation_indexes("relation", successful_indexes, failed_relations)
        self.assertTrue(result)
        self.assertEqual(len(successful_indexes), 2)
        self.assertEqual(len(failed_relations), 0)

    def test_create_relation_indexes_value_error(self):
        self.graph._execute_index_queries = MagicMock(side_effect=ValueError("Invalid value"))
        successful_indexes = []
        failed_relations = []

        result = self.graph._create_relation_indexes("relation", successful_indexes, failed_relations)
        self.assertFalse(result)
        self.assertEqual(len(successful_indexes), 0)
        self.assertEqual(len(failed_relations), 1)

    def test_create_relation_indexes_attribute_error(self):
        self.graph._execute_index_queries = MagicMock(side_effect=AttributeError("Attribute not found"))
        successful_indexes = []
        failed_relations = []

        result = self.graph._create_relation_indexes("relation", successful_indexes, failed_relations)
        self.assertFalse(result)
        self.assertEqual(len(successful_indexes), 0)
        self.assertEqual(len(failed_relations), 1)

    def test_create_relation_indexes_failure(self):
        self.graph._execute_index_queries = MagicMock(side_effect=Exception("Error"))
        successful_indexes = []
        failed_relations = []

        result = self.graph._create_relation_indexes("relation", successful_indexes, failed_relations)
        self.assertFalse(result)
        self.assertEqual(len(successful_indexes), 0)
        self.assertEqual(len(failed_relations), 1)

    def test_execute_index_queries(self):
        mock_cursor = MagicMock()
        self.mock_adapter.get_cursor.return_value.__enter__.return_value = mock_cursor

        self.graph._execute_index_queries(["QUERY1", "QUERY2"])
        mock_cursor.execute.assert_has_calls([call("QUERY1"), call("QUERY2")])
        self.mock_adapter.connection.commit.assert_called_once()

    def test_find_weakly_connected_components(self):
        # Mock the nodes and the query results to simulate a single weakly connected component
        self.graph.get_nodes = MagicMock(return_value=["node1", "node2"])
        self.mock_adapter.execute_cypher_query.side_effect = [
            [{"label": "node2"}],  # node1 is connected to node2
            []  # node2 has no additional connections
        ]

        # Call the method
        components = self.graph._find_weakly_connected_components()

        # Assertions
        self.assertEqual(len(components), 1)
        self.assertIn("node1", components[0])
        self.assertIn("node2", components[0])

    def test_find_strongly_connected_components(self):
        self.graph.get_nodes = MagicMock(return_value=["node1", "node2"])
        self.graph.get_edges = MagicMock(return_value=[("node1", "node2"), ("node2", "node1")])

        components = self.graph._find_strongly_connected_components()
        self.assertEqual(len(components), 1)
        self.assertIn("node1", components[0])
        self.assertIn("node2", components[0])


if __name__ == "__main__":
    unittest.main()
