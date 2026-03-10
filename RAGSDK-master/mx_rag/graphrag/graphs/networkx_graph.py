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

import os
from json import JSONDecodeError
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Callable

import networkx as nx
from loguru import logger
from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.storage.document_store.base_storage import StorageError
from mx_rag.utils.common import check_db_file_limit, write_to_json, read_graph_file
from mx_rag.utils.file_check import check_disk_free_space


class NetworkxGraph(GraphStore):
    """
    Wrapper for NetworkX graphs providing common operations.
    """
    FREE_SPACE_LIMIT = 5 * 1024 * 1024 * 1024  # 5GB

    def __init__(self, is_digraph: bool = True, path: Optional[str] = None, decrypt_fn: Callable = None) -> None:
        """
        Initialize the graph, optionally loading from file.
        """
        self.is_digraph = is_digraph
        self.graph: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if self.is_digraph else nx.Graph()
        if path:
            self._load_graph(path, decrypt_fn)

    def save(self, output_path: str, encrypt_fn: Callable = None):
        """
        Save the graph to a file.
        """
        data = nx.node_link_data(self.graph)
        write_to_json(output_path, data, encrypt_fn)
        logger.info(f"Graph saved to: {output_path}")

    def add_node(self, node: str, **attrs: Any) -> None:
        """
        Add a node with optional attributes.
        """
        if not self.has_node(node):
            self.graph.add_node(node, **attrs)

    def add_nodes_from(self, nodes: Iterable[str], **attrs: Any) -> None:
        """
        Add multiple nodes with shared attributes.
        """
        for node in nodes:
            self.add_node(node, **attrs)

    def remove_node(self, node: str) -> None:
        """
        Remove a node.
        """
        try:
            self.graph.remove_node(node)
        except nx.NetworkXError:
            logger.warning(f"Node '{node}' not found.")

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists.
        """
        return self.graph.has_node(node)

    def get_node_attributes(self, node: str, key: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get attributes of a node, or a specific attribute.
        """
        if not self.has_node(node):
            logger.warning(f"Node '{node}' not found.")
            return {} if key is None else None
        return self.graph.nodes[node].get(key) if key else dict(self.graph.nodes[node])

    def set_node_attributes(self, attributes: Dict[str, Dict[str, Any]], name: str) -> None:
        """
        Set attributes for multiple nodes.
        """
        nx.set_node_attributes(self.graph, attributes, name)

    def update_node_attribute(
            self, node: str, key: str, value: Any, append: bool = False
    ) -> None:
        """
        Update or append a node attribute.
        """
        if self.has_node(node):
            if append and isinstance(self.graph.nodes[node].get(key), str):
                current_value = self.graph.nodes[node].get(key, "")
                values = set(filter(None, map(str.strip, current_value.split(",")))) if current_value else set()
                values.update(map(str.strip, str(value).split(",")))
                self.graph.nodes[node][key] = ",".join(sorted(values))
            else:
                values = set(map(str.strip, str(value).split(",")))
                self.graph.nodes[node][key] = ",".join(values)
        else:
            logger.warning(f"Node '{node}' not found.")

    def update_node_attributes_batch(
            self, node_updates: List[Tuple[str, Dict[str, Any]]], batch_size: int = 1024, append: bool = False
    ) -> None:
        """
        Update attributes for multiple nodes in batch.
        
        Args:
            node_updates: List of (node, attributes_dict) tuples
            batch_size: Not used in this function
            append: Whether to append to existing string attributes
        """
        for node, attrs in node_updates:
            if self.has_node(node):
                for key, value in attrs.items():
                    self.update_node_attribute(node, key, value, append)
            else:
                logger.warning(f"Node '{node}' not found.")

    def get_nodes(self, with_data: bool = True) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        """
        Return all nodes, optionally with attributes.
        """
        return list(self.graph.nodes(data=with_data)) if with_data else list(self.graph.nodes)

    def get_nodes_by_attribute(self, key: str, value: Any) -> List[str]:
        """
        Get nodes where an attribute equals a value.
        """
        return [n for n, data in self.graph.nodes(data=True) if data.get(key) == value]

    def get_nodes_containing_attribute_value(self, key: str, value: str) -> List[str]:
        """
        Get nodes where an attribute contains a substring.
        """
        return [
            n for n, data in self.graph.nodes(data=True)
            if isinstance(data.get(key), str) and value in data.get(key, "")
        ]

    def add_edge(self, u: str, v: str, **attr: Any) -> None:
        """
        Add an edge with optional attributes.
        """
        if not self.has_edge(u, v):
            self.graph.add_edge(u, v, **attr)

    def add_edges_from(self, edges: Iterable[Tuple[str, str, Optional[Dict[str, Any]]]]) -> None:
        """
        Add multiple edges.
        """
        self.graph.add_edges_from(edges)

    def remove_edge(self, u: str, v: str) -> None:
        """
        Remove an edge.
        """
        try:
            self.graph.remove_edge(u, v)
        except nx.NetworkXError:
            logger.warning(f"Edge '{u}', '{v}' not found.")

    def has_edge(self, u: str, v: str) -> bool:
        """
        Check if an edge exists.
        """
        return self.graph.has_edge(u, v)

    def get_edge_attributes(
            self, u: str, v: str, key: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        Get attributes of an edge, or a specific attribute.
        """
        if not self.has_edge(u, v):
            logger.warning(f"Edge '{u}', '{v}' not found.")
            return {} if key is None else None
        return self.graph.edges[u, v].get(key) if key else dict(self.graph.edges[u, v])

    def update_edge_attribute(
            self, u: str, v: str, key: str, value: Any, append: bool = False
    ) -> None:
        """
        Update or append an edge attribute.
        """
        if self.has_edge(u, v):
            if append and isinstance(self.graph.edges[u, v].get(key), str):
                current_value = self.graph.edges[u, v].get(key, "")
                values = set(filter(None, map(str.strip, current_value.split(",")))) if current_value else set()
                values.update(map(str.strip, str(value).split(",")))
                self.graph.edges[u, v][key] = ",".join(sorted(values))
            else:
                values = set(map(str.strip, str(value).split(",")))
                self.graph.edges[u, v][key] = ",".join(values)
        else:
            logger.warning(f"Edge '{u}', '{v}' not found.")

    def update_edge_attributes_batch(
            self, edge_updates: List[Tuple[str, str, Dict[str, Any]]], batch_size: int = 1024, append: bool = False
    ) -> None:
        """
        Update attributes for multiple edges in batch.
        
        Args:
            edge_updates: List of (u, v, attributes_dict) tuples
            batch_size: Not used in this function
            append: Whether to append to existing string attributes
        """
        for u, v, attrs in edge_updates:
            if self.has_edge(u, v):
                for key, value in attrs.items():
                    self.update_edge_attribute(u, v, key, value, append)
            else:
                logger.warning(f"Edge '{u}', '{v}' not found.")

    def get_edges(
            self, with_data: bool = True
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Dict[str, Any]]]]:
        """
        Return all edges, optionally with attributes.
        """
        return list(self.graph.edges(data=with_data)) if with_data else list(self.graph.edges)

    def get_edge_attribute_values(
            self, key: str
    ) -> List[str]:
        """
        Get all values for a specific edge attribute.
        """
        return [
            data[key]
            for _, _, data in self.graph.edges(data=True)
        ]

    def in_degree(self, node: str) -> int:
        """
        Return the in-degree of a node.
        """
        return self.graph.in_degree(node) if self.has_node(node) and self.graph.is_directed() else 0

    def out_degree(self, node: str) -> int:
        """
        Return the out-degree of a node.
        """
        return self.graph.out_degree(node) if self.has_node(node) and self.graph.is_directed() else 0

    def neighbors(self, node: str) -> List[str]:
        """
        Get neighbors of a node.
        For directed graphs, returns successors.
        """
        if not self.has_node(node):
            logger.warning(f"Node '{node}' not found.")
            return []
        return list(self.graph.neighbors(node))

    def successors(self, node: str) -> List[str]:
        """
        Get successors of a node (only for directed graphs).
        """
        if not self.has_node(node):
            logger.warning(f"Node '{node}' not found.")
            return []
        if not self.is_digraph:
            logger.warning("successors() called on undirected graph.")
            return []
        return list(self.graph.successors(node))

    def predecessors(self, node: str) -> List[str]:
        """
        Get predecessors of a node (only for directed graphs).
        """
        if not self.has_node(node):
            logger.warning(f"Node '{node}' not found.")
            return []
        if not self.is_digraph:
            logger.warning("predecessors() called on undirected graph.")
            return []
        return list(self.graph.predecessors(node))

    def number_of_nodes(self) -> int:
        """
        Return the number of nodes.
        """
        return self.graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """
        Return the number of edges.
        """
        return self.graph.number_of_edges()

    def density(self) -> float:
        """
        Return the graph density.
        """
        return nx.density(self.graph)

    def connected_components(self) -> Iterable[Set[str]]:
        """
        Return connected components.
        """
        if self.graph.is_directed():
            return nx.weakly_connected_components(self.graph)
        return nx.connected_components(self.graph)

    def subgraph(self, nodes: Iterable[str]) -> "NetworkxGraph":
        """
        Return a subgraph induced by given nodes.
        """
        subgraph = self.graph.subgraph(nodes).copy()
        return NetworkxGraph(is_digraph=self.graph.is_directed()).set_graph(subgraph)

    def get_subgraph_edges(
            self, nodes: Iterable[str], with_data: bool = True
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Dict[str, Any]]]]:
        """
        Return edges of the subgraph induced by given nodes.
        """
        subgraph = self.graph.subgraph(nodes)
        return list(subgraph.edges(data=with_data)) if with_data else list(subgraph.edges)

    def set_graph(self, graph: Union[nx.Graph, nx.DiGraph]) -> "NetworkxGraph":
        """
        Set the internal graph object.
        """
        self.graph = graph
        return self

    def _load_graph(self, path: str, decrypt_fn: Callable) -> None:
        """
        Load the graph from a file.
        """
        try:
            dirname = os.path.dirname(path)
            if check_disk_free_space(dirname if dirname else "./", self.FREE_SPACE_LIMIT):
                raise StorageError("Insufficient remaining space, please clear disk space")
            check_db_file_limit(path)
            data = read_graph_file(path, decrypt_fn)
            self.graph = nx.node_link_graph(data)
            logger.info(f"Graph loaded from: {path}")
        except (TypeError, ValueError, JSONDecodeError):
            logger.warning(f"Failed to load graph: {path}, invalid json format.")
        except FileNotFoundError:
            logger.warning(f"Graph file not found at: {path}. Creating an empty graph.")
        except Exception as e:
            logger.warning(f"Error loading graph from {path}: {e}")
