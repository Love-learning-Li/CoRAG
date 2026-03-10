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
import networkx as nx

from mx_rag.graphrag.graphs.networkx_graph import NetworkxGraph


class TestNetworkxGraph(unittest.TestCase):
    def setUp(self):
        self.graph = NetworkxGraph(is_digraph=True)

    def test_add_and_has_node(self):
        self.graph.add_node("A")
        self.assertTrue(self.graph.has_node("A"))
        self.assertFalse(self.graph.has_node("B"))

    def test_add_node_with_attributes(self):
        self.graph.add_node("A", color="red", value=1)
        attrs = self.graph.get_node_attributes("A")
        self.assertEqual(attrs["color"], "red")
        self.assertEqual(attrs["value"], 1)

    def test_add_nodes_from(self):
        self.graph.add_nodes_from(["A", "B", "C"], group="test")
        nodes = self.graph.get_nodes()
        node_names = [n for n, _ in nodes]
        self.assertIn("A", node_names)
        self.assertIn("B", node_names)
        self.assertIn("C", node_names)
        for _, attrs in nodes:
            self.assertEqual(attrs["group"], "test")

    def test_remove_node(self):
        self.graph.add_node("A")
        self.graph.remove_node("A")
        self.assertFalse(self.graph.has_node("A"))

    def test_get_node_attributes_specific_key(self):
        self.graph.add_node("A", foo="bar", num=5)
        self.assertEqual(self.graph.get_node_attributes("A", "foo"), "bar")
        self.assertEqual(self.graph.get_node_attributes("A", "num"), 5)
        self.assertIsNone(self.graph.get_node_attributes("A", "not_exist"))

    def test_set_node_attributes(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.set_node_attributes({"A": "blue", "B": "green"}, "color")
        self.assertEqual(self.graph.get_node_attributes("A", "color"), "blue")
        self.assertEqual(self.graph.get_node_attributes("B", "color"), "green")

    def test_update_node_attribute_append(self):
        self.graph.add_node("A", tags="x,y")
        self.graph.update_node_attribute("A", "tags", "z", append=True)
        tags = self.graph.get_node_attributes("A", "tags")
        self.assertIn("x", tags)
        self.assertIn("y", tags)
        self.assertIn("z", tags)
        self.assertEqual(set(tags.split(",")), {"x", "y", "z"})

    def test_update_node_attribute_overwrite(self):
        self.graph.add_node("A", tags="x,y")
        self.graph.update_node_attribute("A", "tags", "z", append=False)
        tags = self.graph.get_node_attributes("A", "tags")
        self.assertEqual(tags, "z")

    def test_update_node_attributes_batch(self):
        self.graph.add_nodes_from(["A", "B"])
        updates = [("A", {"foo": "bar"}), ("B", {"foo": "baz"})]
        self.graph.update_node_attributes_batch(updates)
        self.assertEqual(self.graph.get_node_attributes("A", "foo"), "bar")
        self.assertEqual(self.graph.get_node_attributes("B", "foo"), "baz")

    def test_get_nodes_by_attribute(self):
        self.graph.add_node("A", color="red")
        self.graph.add_node("B", color="blue")
        self.assertEqual(self.graph.get_nodes_by_attribute("color", "red"), ["A"])
        self.assertEqual(self.graph.get_nodes_by_attribute("color", "blue"), ["B"])

    def test_get_nodes_containing_attribute_value(self):
        self.graph.add_node("A", tags="foo,bar")
        self.graph.add_node("B", tags="baz,qux")
        result = self.graph.get_nodes_containing_attribute_value("tags", "foo")
        self.assertIn("A", result)
        self.assertNotIn("B", result)

    def test_add_and_has_edge(self):
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B", weight=2)
        self.assertTrue(self.graph.has_edge("A", "B"))
        self.assertFalse(self.graph.has_edge("B", "A"))  # Directed

    def test_add_edges_from(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from([("A", "B"), ("B", "C")])
        self.assertTrue(self.graph.has_edge("A", "B"))
        self.assertTrue(self.graph.has_edge("B", "C"))

    def test_remove_edge(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B")
        self.graph.remove_edge("A", "B")
        self.assertFalse(self.graph.has_edge("A", "B"))

    def test_get_edge_attributes(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B", weight=5, label="foo")
        self.assertEqual(self.graph.get_edge_attributes("A", "B", "weight"), 5)
        self.assertEqual(self.graph.get_edge_attributes("A", "B", "label"), "foo")
        self.assertIsNone(self.graph.get_edge_attributes("A", "B", "not_exist"))

    def test_update_edge_attribute_append(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B", tags="x,y")
        self.graph.update_edge_attribute("A", "B", "tags", "z", append=True)
        tags = self.graph.get_edge_attributes("A", "B", "tags")
        self.assertEqual(set(tags.split(",")), {"x", "y", "z"})

    def test_update_edge_attribute_overwrite(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B", tags="x,y")
        self.graph.update_edge_attribute("A", "B", "tags", "z", append=False)
        tags = self.graph.get_edge_attributes("A", "B", "tags")
        self.assertEqual(tags, "z")

    def test_update_edge_attributes_batch(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from([("A", "B"), ("B", "C")])
        updates = [("A", "B", {"foo": "bar"}), ("B", "C", {"foo": "baz"})]
        self.graph.update_edge_attributes_batch(updates)
        self.assertEqual(self.graph.get_edge_attributes("A", "B", "foo"), "bar")
        self.assertEqual(self.graph.get_edge_attributes("B", "C", "foo"), "baz")

    def test_get_edges(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B", weight=1)
        edges = self.graph.get_edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][0], "A")
        self.assertEqual(edges[0][1], "B")
        self.assertEqual(edges[0][2]["weight"], 1)

    def test_get_edge_attribute_values(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edge("A", "B", label="foo")
        self.graph.add_edge("B", "C", label="bar")
        values = self.graph.get_edge_attribute_values("label")
        self.assertIn("foo", values)
        self.assertIn("bar", values)

    def test_in_out_degree(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B")
        self.assertEqual(self.graph.in_degree("A"), 0)
        self.assertEqual(self.graph.out_degree("A"), 1)
        self.assertEqual(self.graph.in_degree("B"), 1)
        self.assertEqual(self.graph.out_degree("B"), 0)

    def test_neighbors_successors_predecessors(self):
        self.graph.add_nodes_from(["A", "B"])
        self.graph.add_edge("A", "B")
        self.assertIn("B", self.graph.neighbors("A"))
        self.assertIn("B", self.graph.successors("A"))
        self.assertIn("A", self.graph.predecessors("B"))

    def test_number_of_nodes_and_edges(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from([("A", "B"), ("B", "C")])
        self.assertEqual(self.graph.number_of_nodes(), 3)
        self.assertEqual(self.graph.number_of_edges(), 2)

    def test_density(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from([("A", "B"), ("B", "C")])
        density = self.graph.density()
        density_is_valid = 0 < density < 1
        self.assertTrue(density_is_valid)

    def test_connected_components(self):
        g = NetworkxGraph(is_digraph=False)
        g.add_nodes_from(["A", "B", "C", "D"])
        g.add_edges_from([("A", "B"), ("C", "D")])
        components = list(g.connected_components())
        self.assertEqual(len(components), 2)
        all_nodes = set().union(*components)
        self.assertEqual(all_nodes, {"A", "B", "C", "D"})

    def test_subgraph_and_get_subgraph_edges(self):
        self.graph.add_nodes_from(["A", "B", "C"])
        self.graph.add_edges_from([("A", "B"), ("B", "C")])
        sub = self.graph.subgraph(["A", "B"])
        self.assertTrue(sub.has_node("A"))
        self.assertTrue(sub.has_node("B"))
        self.assertFalse(sub.has_node("C"))
        edges = self.graph.get_subgraph_edges(["A", "B"])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][0], "A")
        self.assertEqual(edges[0][1], "B")

    def test_set_graph(self):
        g = nx.DiGraph()
        g.add_node("X")
        g.add_edge("X", "Y")
        self.graph.set_graph(g)
        self.assertTrue(self.graph.has_node("X"))
        self.assertTrue(self.graph.has_edge("X", "Y"))

    def test_load_graph_file_not_found(self):
        path = "nonexistent_file.json"
        g = NetworkxGraph(is_digraph=True, path=path)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)
