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
import re
import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Callable
from typing import Any

import networkx as nx
from langchain_opengauss import openGaussAGEGraph
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.graphrag.graphs.graph_util import OpenGaussAGEAdapter, CypherQueryBuilder, cypher_value
from mx_rag.utils.common import validate_params, write_to_json


class OpenGaussGraph(GraphStore):
    """
    Adapter for openGaussAGEGraph, providing a NetworkX-like interface for graph operations.
    """

    def __init__(self, graph_name: str, age_graph: openGaussAGEGraph):
        """
        Initialize an OpenGaussGraph instance.

        Args:
            graph_name: Name of the graph in the database.
            conf: OpenGaussSettings configuration object.
        """
        if not graph_name.isidentifier():
            raise ValueError(f"Invalid graph name: {graph_name}")

        self.graph_name = graph_name
        self.age_graph = age_graph
        self.graph_adapter = OpenGaussAGEAdapter(age_graph)

    def add_node(self, node: str, **attributes: Any) -> None:
        """
        Add a node with optional attributes.
        """
        if self.has_node(node):
            return
        label = hashlib.sha256(node.encode("utf-8")).hexdigest()
        attributes.update({"text": node, 'id': label})
        query = CypherQueryBuilder.merge_node(attributes)
        self.graph_adapter.execute_cypher_query(query)

    def save(self, output_path: str, encrypt_fn: Callable = None) -> None:
        """
        Export the graph's nodes and edges in node-link format to a JSON file.

        Args:
            encrypt_fn: function to encrypt text
            output_path: Path to the output JSON file.
        """
        # Get all nodes with data
        nodes = self.get_nodes(with_data=True)
        node_list = []
        for _, (label, props) in enumerate(nodes):
            node_entry = {"id": label}
            node_entry.update(props)
            node_list.append(node_entry)

        # Get all edges with data
        edges = []
        edge_data = self.get_edges(with_data=True)
        if not edge_data:
            # fallback: try to get all edges without attributes
            query = "MATCH (a)-[r]->(b) RETURN a.id AS u, b.id AS v"
            result = self.graph_adapter.execute_cypher_query(query)
            for row in result:
                edges.append({"source": row["u"], "target": row["v"]})
        else:
            for u, v, props in edge_data:
                edge_entry = {"source": u, "target": v}
                if props:
                    edge_entry.update(props)
                edges.append(edge_entry)

        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": node_list,
            "links": edges,
        }
        write_to_json(output_path, graph_data, encrypt_fn)
        logger.info(f"Graph saved to: {output_path}")

    @validate_params(nodes=dict(validator=lambda x: len(x) < 10000, message="Node label list is too long."))
    def add_nodes_from(
            self, nodes: Iterable[Union[str, Tuple[str, Dict[str, Any]]]], **common_attrs: Any
    ) -> None:
        """
        Add multiple nodes, optionally with attributes.

        Args:
            nodes: Iterable of node labels or (label, attributes) tuples.
            **common_attrs: Attributes applied to all nodes.
        """
        for node in nodes:
            attrs = common_attrs.copy()
            if isinstance(node, tuple):
                node, node_attrs = node
                attrs.update(node_attrs)
            if not self.has_node(node):
                self.add_node(node, **attrs)

    def remove_node(self, node: str) -> None:
        """Remove a node."""
        if self.has_node(node):
            query = CypherQueryBuilder.delete_node(hashlib.sha256(node.encode("utf-8")).hexdigest())
            self.graph_adapter.execute_cypher_query(query)

    def has_node(self, node: str) -> bool:
        """
        Check if a node exists.
        """
        node_id = hashlib.sha256(node.encode("utf-8")).hexdigest()
        query = CypherQueryBuilder.match_node(node_id)
        try:
            result = self.graph_adapter.execute_cypher_query(query)
            return bool(result)
        except (TypeError, AttributeError):
            return False
        except Exception as e:
            error_msg = str(e).lower()
            # Handle various database error messages for non-existent labels/nodes
            if any(phrase in error_msg for phrase in ["does not exist", "not found", "label does not exist"]):
                return False
            logger.warning(f"Unexpected error checking node existence for '{node}': {e}")
            raise

    def get_node_attributes(self, node: str, key: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """
        Retrieve node attributes or a specific attribute.

        Args:
            node: Node label.
            key: Attribute key (optional).

        Returns:
            Dict of attributes, a single value, or None.
        """
        label = hashlib.sha256(node.encode("utf-8")).hexdigest()
        if key is None:
            query = CypherQueryBuilder.match_node_properties(label)
            result = self.graph_adapter.execute_cypher_query(query)
            return result[0]['props'] if result else None
        query = CypherQueryBuilder.match_node_attribute(label, key)
        result = self.graph_adapter.execute_cypher_query(query)
        return result[0]['value'] if result else None

    def set_node_attributes(self, attributes: Dict[Any, Any], name: str) -> None:
        """
        Set node attributes for multiple nodes.

        Args:
            attributes: Dict mapping node labels to attribute values.
            name: Attribute name to set.
        """
        if not attributes:
            return
        props = [
            {"label": hashlib.sha256(str(label).encode("utf-8")).hexdigest(), "value": value}
            for label, value in attributes.items()
        ]
        # Convert props to Cypher list syntax
        cypher_list = "[" + ", ".join(
            "{" + f'label: "{item["label"]}", value: {json.dumps(item["value"])}' + "}" for item in props
        ) + "]"
        query = CypherQueryBuilder.set_node_attributes(name, cypher_list)
        self.graph_adapter.execute_cypher_query(query)

    def update_node_attribute(self, node: str, key: str, value: Any, append: bool = False) -> None:
        """
        Update or append a node attribute.

        Args:
            node: Node label.
            key: Attribute key.
            value: New value.
            append: If True, append to a list attribute.
        """
        label = hashlib.sha256(node.encode("utf-8")).hexdigest()
        if append:
            old_value = self.get_node_attributes(node, key)
            if isinstance(old_value, str) and value not in old_value.split(","):
                value = ",".join(filter(None, [old_value, value]))
            else:
                value = value if old_value is None else old_value
        query = CypherQueryBuilder.set_node_attribute(label, key, value, False)
        self.graph_adapter.execute_cypher_query(query)

    @validate_params(node_updates=dict(validator=lambda x: len(x) < 10000, message="Node list is too long."))
    def update_node_attributes_batch(self, node_updates: list, batch_size=2048, append: bool = False) -> None:
        """
        Batch update node attributes with deduplication and merging support.
        
        Efficiently updates multiple nodes' attributes in batches, with support for
        appending and deduplicating comma-separated values.
        
        Args:
            node_updates (list): List of (node_label, attributes_dict) tuples.
                Each tuple contains:
                - node_label (str): The node label
                - attributes_dict (Dict[str, Any]): Dictionary of attribute key-value pairs to update
            batch_size (int, optional): Number of node updates to process per batch. Defaults to 2048.
            append (bool, optional): If True, merges new values with existing values.
                If False, only processes the provided new values. Defaults to False.
        
        Returns:
            None
            
        Note:
            - Node labels are automatically hashed to hashlib.sha256 for internal storage
            - Attribute values are treated as comma-separated strings for deduplication
            - Empty values are filtered out during the merge process
        """
        if not node_updates:
            return

        update_keys = set()
        for _, attrs in node_updates:
            update_keys.update(attrs.keys())
        update_keys = list(update_keys)

        # Fetch current values for all nodes and keys
        node_ids = [hashlib.sha256(str(node_label).encode("utf-8")).hexdigest() for node_label, _ in node_updates]
        current_attrs = {}
        if append:
            current_attrs = {nid: self.get_node_attributes(node_label) or {}
                             for node_label, nid in zip([n for n, _ in node_updates], node_ids)}

        unwind_data = []
        for (node_label, attrs), nid in zip(node_updates, node_ids):
            node_dict = {"id": nid}
            current = current_attrs.get(nid, {})
            for k in update_keys:
                old_val = current.get(k, "")
                new_val = attrs.get(k, "")
                # Split, deduplicate, and join
                old_set = set(filter(None, map(str.strip, old_val.split(",")))) if old_val else set()
                new_set = set(filter(None, map(str.strip, new_val.split(",")))) if new_val else set()
                merged = ",".join(sorted(old_set | new_set)) if (old_set or new_set) else ""
                node_dict[k] = merged
            unwind_data.append(node_dict)

        set_clause = ", ".join([f"n.{k} = item.{k}" for k in update_keys])
        # Batch processing
        total_batches = (len(unwind_data) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(unwind_data), batch_size), total=total_batches, desc="Updating node attributes"):
            batch = unwind_data[i:i + batch_size]
            cypher_list = "[" + ", ".join(
                "{" + ", ".join(f"{k}: {json.dumps(v)}" for k, v in d.items()) + "}"
                for d in batch
            ) + "]"
            query = (
                f"UNWIND {cypher_list} AS item "
                f"MATCH (n:Node) WHERE n.id = item.id "
                f"SET {set_clause}"
            )
            self.graph_adapter.execute_cypher_query(query)
        logger.info("Successfully updated node attributes")

    @validate_params(edge_updates=dict(validator=lambda x: len(x) < 10000, message="Edge updates are too long."))
    def update_edge_attributes_batch(self, edge_updates: list, batch_size: int = 2048, append: bool = False) -> None:
        """
        Batch update edge attributes with deduplication and merging support.
        
        This method efficiently updates multiple edges' attributes in batches, with support for
        appending and deduplicating comma-separated values. For each edge, it merges old and new
        attribute values by splitting on commas, removing duplicates, and rejoining.
        
        Args:
            edge_updates (list): List of (source_node, target_node, attributes_dict) tuples.
                Each tuple contains:
                - source_node (str): The source node label
                - target_node (str): The target node label  
                - attributes_dict (Dict[str, Any]): Dictionary of attribute key-value pairs to update
            batch_size (int, optional): Number of edge updates to process per Cypher query batch.
                Larger batches improve performance but use more memory. Defaults to 2048.
            append (bool, optional): If True, fetches current attribute values and merges them
                with new values. If False, only processes the provided new values. Defaults to False.
        
        Returns:
            None
            
        Raises:
            Exception: If Cypher query execution fails during batch processing
        """
        logger.debug("Updating edge attributes...")
        if not edge_updates:
            return

        update_keys = set()
        for _, _, attrs in edge_updates:
            update_keys.update(attrs.keys())
        update_keys = list(update_keys)

        # Fetch current values for all edges and keys
        edge_ids = [
            (
                hashlib.sha256(str(u).encode("utf-8")).hexdigest(),
                hashlib.sha256(str(v).encode("utf-8")).hexdigest()
            ) for u, v, _ in edge_updates
        ]
        current_attrs = {}
        if append:
            current_attrs = {
                (uid, vid): self.get_edge_attributes(u, v) or {}
                for (u, v, _), (uid, vid) in zip(edge_updates, edge_ids)
            }

        unwind_data = []
        for (_, _, attrs), (uid, vid) in tqdm(
                zip(edge_updates, edge_ids), total=len(edge_ids), desc="Preparing unwind data"
        ):
            edge_dict = {"start_id": uid, "end_id": vid}
            current = current_attrs.get((uid, vid), {})
            for k in update_keys:
                old_val = current.get(k, "")
                new_val = attrs.get(k, "")
                old_set = set(filter(None, map(str.strip, old_val.split(",")))) if old_val else set()
                new_set = set(filter(None, map(str.strip, new_val.split(",")))) if new_val else set()
                merged = ",".join(sorted(old_set | new_set)) if (old_set or new_set) else ""
                edge_dict[k] = merged
            unwind_data.append(edge_dict)

        set_clause = ", ".join([f"r.{k} = item.{k}" for k in update_keys])
        # Batch processing
        total_batches = (len(unwind_data) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(unwind_data), batch_size), total=total_batches, desc="Updating edge attributes"):
            batch = unwind_data[i:i + batch_size]
            cypher_list = "[" + ", ".join(
                "{" + ", ".join(f"{k}: {json.dumps(v)}" for k, v in d.items()) + "}"
                for d in batch
            ) + "]"
            query = (
                f"UNWIND {cypher_list} AS item "
                f"MATCH (a:Node)-[r]->(b:Node) "
                f"WHERE a.id = item.start_id AND b.id = item.end_id "
                f"SET {set_clause}"
            )
            self.graph_adapter.execute_cypher_query(query)
        logger.info("Successfully updated edge attributes.")

    def get_nodes(self, with_data: bool = True) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        """
        Retrieve all nodes.

        Args:
            with_data: If True, return attributes as well.

        Returns:
            List of node labels or (label, attributes) tuples.
        """
        query = CypherQueryBuilder.match_nodes(with_data)
        result = self.graph_adapter.execute_cypher_query(query)
        if with_data:
            return [(row['label'], row['props']) for row in result]
        return [row['label'] for row in result]

    def get_nodes_by_attribute(self, key: str, value: Any) -> List[str]:
        """
        Retrieve nodes with a specific attribute value.

        Args:
            key: Attribute key.
            value: Attribute value.

        Returns:
            List of node labels.
        """
        query = CypherQueryBuilder.match_nodes_by_attribute(key, value)
        result = self.graph_adapter.execute_cypher_query(query)
        return [row["props"]["text"] for row in result]

    def get_nodes_containing_attribute_value(self, key: str, value: str) -> List[str]:
        """
        Retrieve nodes where the attribute contains a substring.

        Args:
            key: Attribute key.
            value: Substring to search for.

        Returns:
            List of node labels.
        """
        query = CypherQueryBuilder.match_nodes_containing_attribute(key, value)
        result = self.graph_adapter.execute_cypher_query(query)
        return [row["props"]["text"] for row in result]

    def add_edge(self, u: str, v: str, **attributes: Any) -> None:
        """
        Add an edge between two nodes, ensuring both nodes exist and indexes are created for efficient queries.
        """
        # Ensure both nodes exist (idempotent)
        for node in (u, v):
            if not self.has_node(node):
                self.add_node(node)

        if self.has_edge(u, v):
            return

        # Hash node labels for internal use
        u_hashed = hashlib.sha256(u.encode("utf-8")).hexdigest()
        v_hashed = hashlib.sha256(v.encode("utf-8")).hexdigest()

        # Build and execute the Cypher query for the edge
        query = CypherQueryBuilder.merge_edge(u_hashed, v_hashed, attributes)
        try:
            self.graph_adapter.execute_cypher_query(query)
        except SyntaxError as e:
            logger.error(f"Cypher query syntax error: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to add edge from '{u}' to '{v}': {e}")
            raise

    @validate_params(edges=dict(validator=lambda x: len(x) < 10000, message="Too many edges."))
    def add_edges_from(
            self,
            edges: Iterable[Union[Tuple[str, str], Tuple[str, str, Optional[Dict[str, Any]]]]]
    ) -> None:
        """
        Add multiple edges.

        Args:
            edges: Iterable of (source_label, target_label) or (source_label, target_label, attributes) tuples.
        """
        for edge in edges:
            if len(edge) == 3:
                source, target, attrs = edge
                self.add_edge(source, target, **(attrs or {}))
            else:
                source, target = edge
                self.add_edge(source, target)

    def remove_edge(self, u: str, v: str) -> None:
        """
        Remove an edge between two nodes.

        Args:
            u: Source node label.
            v: Target node label.
        """
        if self.has_edge(u, v):
            u = hashlib.sha256(u.encode("utf-8")).hexdigest()
            v = hashlib.sha256(v.encode("utf-8")).hexdigest()
            query = CypherQueryBuilder.delete_edge(u, v)
            self.graph_adapter.execute_cypher_query(query)

    def has_edge(self, u: str, v: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            u: Source node label.
            v: Target node label.

        Returns:
            True if edge exists, False otherwise.
        """
        u = hashlib.sha256(u.encode("utf-8")).hexdigest()
        v = hashlib.sha256(v.encode("utf-8")).hexdigest()
        query = CypherQueryBuilder.match_edge(u, v)
        try:
            result = self.graph_adapter.execute_cypher_query(query)
            return bool(result)
        except SyntaxError as e:
            logger.error(f"Cypher query syntax error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid value encountered: {e}")
            raise
        except Exception as e:
            # Handle the case where the label does not exist
            if "does not exist" in str(e):
                return False
            raise

    def get_edge_attributes(
            self, u: str, v: str, key: Optional[str] = None
    ) -> Union[Dict[str, Any], Any, None]:
        """
        Retrieve edge attributes or a specific attribute.

        Args:
            u: Source node label.
            v: Target node label.
            key: Specific attribute key (optional).

        Returns:
            Dict of attributes, a single value, or None.
        """
        u = hashlib.sha256(u.encode("utf-8")).hexdigest()
        v = hashlib.sha256(v.encode("utf-8")).hexdigest()
        query = CypherQueryBuilder.match_edge_attribute(u, v, key)
        result = self.graph_adapter.execute_cypher_query(query)
        if key:
            return result[0]['value'] if result else None
        return result[0]['props'] if result else None

    def update_edge_attribute(
            self, u: str, v: str, key: str, value: Any, append: bool = False
    ) -> None:
        """
        Update or append an edge attribute.

        Args:
            u: Source node label.
            v: Target node label.
            key: Attribute key.
            value: New value.
            append: If True, append to a list attribute.
        """
        if self.has_edge(u, v):
            u = hashlib.sha256(u.encode("utf-8")).hexdigest()
            v = hashlib.sha256(v.encode("utf-8")).hexdigest()
            query = CypherQueryBuilder.set_edge_attribute(u, v, key, value, append)
            self.graph_adapter.execute_cypher_query(query)

    def get_edges(
            self, with_data: bool = True
    ) -> Union[List[Tuple[Any, Any]], List[Tuple[Any, Any, Dict[str, Any]]]]:
        """
        Return all edges, optionally with attributes.
        """
        query = CypherQueryBuilder.match_edges(with_data)
        result = self.graph_adapter.execute_cypher_query(query)
        if with_data:
            return [
                (
                    row["source"],
                    row["target"],
                    {**row.get("props", {}), "start_id": row["start_id"], "end_id": row["end_id"]},
                )
                for row in result
            ]
        else:
            return [
                (
                    row["source"],
                    row["target"],
                    {"start_id": row["start_id"], "end_id": row["end_id"]},
                )
                for row in result
            ]

    def get_edge_attribute_values(self, key: str) -> List[str]:
        """
        Get all values for a specific edge attribute.

        Args:
            key: Attribute key.

        Returns:
            List of (source, target, attributes) tuples.
        """
        query = CypherQueryBuilder.match_edges_by_attribute(key)
        result = self.graph_adapter.execute_cypher_query(query)
        return [row['props'][key] for row in result]

    def in_degree(self, node: str) -> int:
        """
        Get the in-degree of a node.

        Args:
            node: Node label.

        Returns:
            In-degree count.
        """
        query = CypherQueryBuilder.in_degree(hashlib.sha256(node.encode("utf-8")).hexdigest())
        result = self.graph_adapter.execute_cypher_query(query)
        return result[0]['deg'] if result else 0

    def out_degree(self, node: str) -> int:
        """
        Get the out-degree of a node.

        Args:
            node: Node label.

        Returns:
            Out-degree count.
        """
        query = CypherQueryBuilder.out_degree(hashlib.sha256(node.encode("utf-8")).hexdigest())
        result = self.graph_adapter.execute_cypher_query(query)
        return result[0]['deg'] if result else 0

    def neighbors(self, node: str) -> List[str]:
        """
        Get all neighbors of a node.

        Args:
            node: Node label.

        Returns:
            List of neighbor node labels.
        """
        query = CypherQueryBuilder.neighbors(hashlib.sha256(node.encode("utf-8")).hexdigest())
        result = self.graph_adapter.execute_cypher_query(query)
        return [row['label'] for row in result]

    def successors(self, node: str) -> List[str]:
        """
        Get all successors of a node.

        Args:
            node: Node label.

        Returns:
            List of successor node labels.
        """
        query = CypherQueryBuilder.successors(hashlib.sha256(node.encode("utf-8")).hexdigest())
        result = self.graph_adapter.execute_cypher_query(query)
        return [row['label'] for row in result]

    def predecessors(self, node: str) -> List[str]:
        """
        Get all predecessors of a node.

        Args:
            node: Node label.

        Returns:
            List of predecessor node labels.
        """
        query = CypherQueryBuilder.predecessors(hashlib.sha256(node.encode("utf-8")).hexdigest())
        result = self.graph_adapter.execute_cypher_query(query)
        return [row['label'] for row in result]

    def number_of_nodes(self) -> int:
        """
        Get the total number of nodes.

        Returns:
            Node count.
        """
        query = CypherQueryBuilder.count_nodes()
        result = self.graph_adapter.execute_cypher_query(query)
        return result[0]['cnt'] if result else 0

    def number_of_edges(self) -> int:
        """
        Get the total number of edges.

        Returns:
            Edge count.
        """
        query = CypherQueryBuilder.count_edges()
        result = self.graph_adapter.execute_cypher_query(query)
        return result[0]['cnt'] if result else 0

    def density(self) -> float:
        """
        Calculate the density of the graph.

        Returns:
            Graph density.
        """
        n = self.number_of_nodes()
        m = self.number_of_edges()
        return m / (n * (n - 1)) if n > 1 else 0.0

    def connected_components(self, is_directed: bool = True) -> List[Set[Any]]:
        """
        Find connected components in the graph.

        Args:
            is_directed: If True, find weakly connected components; otherwise, strongly connected.

        Returns:
            List of sets, each containing node labels in a component.
        """
        if is_directed:
            return self._find_weakly_connected_components()
        else:
            return self._find_strongly_connected_components()

    def create_index_for_edge(self):
        """
        Efficiently create indexes on the start_id and end_id columns for a given edge relation.
        Ensures index names are safe and logs actions for traceability.
        """
        relations = set(self.get_edge_attribute_values("relation"))
        if not relations:
            logger.info("No edge relations found, skipping index creation")
            return

        successful_indexes = []
        failed_relations = []

        for relation in relations:
            success = self._create_relation_indexes(relation, successful_indexes, failed_relations)
            if not success:
                logger.warning(f"Failed to create indexes for relation '{relation}'")

        logger.info(f"Index creation completed. Success: {len(successful_indexes)}, Failed: {len(failed_relations)}")

    def create_index_for_node(self):
        """
        Efficiently create indexes on the Node label for id and properties fields.
        Handles CREATE INDEX CONCURRENTLY outside of a transaction block.
        """
        index_definitions = self._get_node_index_definitions()

        for index_name, query, is_concurrent in index_definitions:
            success = self._create_single_node_index(query, is_concurrent, index_name)
            if not success:
                logger.warning(f"Failed to create index: {index_name}")

    def reset(self) -> None:
        """
        Efficiently reset the graph: drop and recreate all nodes and edges, with logging.
        """
        # Drop the entire graph (including all nodes/edges/labels)
        with self.graph_adapter.get_cursor() as cursor:
            cursor.execute("SELECT * FROM ag_catalog.drop_graph(%s, true)", (self.graph_name,))
            self.graph_adapter.connection.commit()
        logger.info(f"Graph '{self.graph_name}' has been reset, please reconnect.")

    @validate_params(nodes=dict(validator=lambda x: len(x) < 10000, message="Too many nodes."))
    def subgraph(self, nodes: Iterable[str], depth: int = 2) -> list:
        """
        Get all edges in the subgraph induced by the given nodes and all nodes/edges reachable within depth 2.

        Args:
            nodes: Node labels.
            depth: The max depth to traverse.

        Returns:
            List of (source, target, attributes) tuples for all edges in the subgraph.
        """
        # Hash node labels for internal use
        label_list = [hashlib.sha256(node.encode("utf-8")).hexdigest() for node in nodes]
        query = (
            f"UNWIND {cypher_value(label_list)} AS node_id "
            f"MATCH (start:Node {{id: node_id}}) "
            f"MATCH (start)-[*0..{depth}]-(other:Node) "
            f"WITH collect(DISTINCT start.id) + collect(DISTINCT other.id) AS subgraph_ids "
            f"MATCH (a:Node)-[r]->(b:Node) "
            f"WHERE a.id IN subgraph_ids AND b.id IN subgraph_ids "
            f"RETURN a.text AS source, r.relation AS relation, b.text AS target"
        )
        edge_rows = self.graph_adapter.execute_cypher_query(query)
        if not edge_rows:
            return []

        return [(row['source'], row['relation'], row['target']) for row in edge_rows]

    def get_subgraph_edges(
            self, nodes: Iterable[str], with_data: bool = True
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Dict[str, Any]]]]:
        raise NotImplementedError

    def _get_node_index_definitions(self) -> List[Tuple[str, str, bool]]:
        """Get all node index definitions with their configurations."""
        return [
            (
                "index_graph_node_id",
                f"CREATE INDEX IF NOT EXISTS index_graph_node_id ON {self.graph_name}.\"Node\"(id);",
                False
            ),
            (
                "index_graph_node_prop",
                f"CREATE INDEX IF NOT EXISTS index_graph_node_prop ON {self.graph_name}.\"Node\" "
                f"USING gin (properties);",
                False
            ),
            (
                "index_graph_node_id_prop",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS index_graph_node_id_prop ON {self.graph_name}.\"Node\""
                f"(ag_catalog.agtype_access_operator(properties, '\"id\"'::agtype));",
                True
            ),
        ]

    def _create_single_node_index(self, query: str, is_concurrent: bool, index_name: str) -> bool:
        """Create a single node index. Returns True if successful."""
        try:
            if is_concurrent:
                self._execute_concurrent_index(query)
            else:
                self._execute_regular_index(query)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in query: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid value encountered: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}': {e}")
            return False

    def _execute_concurrent_index(self, query: str) -> None:
        """Execute a concurrent index creation outside transaction block."""
        conn = self.graph_adapter.connection
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as curs:
                curs.execute(query)
        finally:
            conn.autocommit = old_autocommit

    def _execute_regular_index(self, query: str) -> None:
        """Execute a regular index creation within transaction block."""
        try:
            with self.graph_adapter.get_cursor() as curs:
                curs.execute(query)
            self.graph_adapter.connection.commit()
        except SyntaxError as e:
            logger.error(f"Cypher query syntax error: {e}")
            self.graph_adapter.connection.rollback()
            raise
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            self.graph_adapter.connection.rollback()
            raise
        except Exception as e:
            self.graph_adapter.connection.rollback()
            raise e

    def _generate_safe_index_prefix(self, relation: str) -> str:
        """Generate a safe and unique index prefix for a relation."""
        safe_prefix = re.sub(r'\W|^(?=\d)', '_', relation.lower())
        hash_suffix = hashlib.sha256(relation.encode('utf-8')).hexdigest()[:8]
        return f"{safe_prefix}_{hash_suffix}"

    def _build_index_queries(self, relation: str, index_prefix: str) -> List[str]:
        """Build the SQL queries for creating start_id and end_id indexes."""
        return [
            f"CREATE INDEX IF NOT EXISTS index_{index_prefix}_start ON "
            f"{self.graph_name}.\"{cypher_value(relation)}\"(start_id);",
            f"CREATE INDEX IF NOT EXISTS index_{index_prefix}_end ON "
            f"{self.graph_name}.\"{cypher_value(relation)}\"(end_id);"
        ]

    def _create_relation_indexes(
            self,
            relation: str,
            successful_indexes: List[str],
            failed_relations: List[str]
    ) -> bool:
        """Create indexes for a single relation. Returns True if successful."""
        index_prefix = self._generate_safe_index_prefix(relation)
        index_queries = self._build_index_queries(relation, index_prefix)

        try:
            self._execute_index_queries(index_queries)
            successful_indexes.extend([f"index_{index_prefix}_start", f"index_{index_prefix}_end"])
            return True
        except ValueError as e:
            logger.error(f"Invalid value encountered: {e}")
            self.graph_adapter.connection.rollback()
            failed_relations.append(relation)
            return False
        except AttributeError as e:
            logger.error(f"Attribute error: {e}")
            self.graph_adapter.connection.rollback()
            failed_relations.append(relation)
            return False
        except Exception as e:
            self.graph_adapter.connection.rollback()
            logger.error(f"Failed to create index for relation '{relation}': {e}")
            failed_relations.append(relation)
            return False

    def _execute_index_queries(self, queries: List[str]) -> None:
        """Execute a list of index creation queries within a transaction."""
        with self.graph_adapter.get_cursor() as curs:
            for query in queries:
                curs.execute(query)
            self.graph_adapter.connection.commit()

    def _find_weakly_connected_components(self) -> List[Set[str]]:
        """
        Find all weakly connected components in the graph using Cypher queries.

        Returns:
            List of sets, each containing node labels in a component.
        """
        all_nodes = list(dict.fromkeys(self.get_nodes(with_data=False)))
        visited = set()
        components = []

        for node in all_nodes:
            if node in visited:
                continue
            # Find all nodes connected to 'node' (ignoring direction)
            query = (
                f"MATCH (start:Node {{id: \"{node}\"}}) "
                f"MATCH p = (start)-[*]-(n) "
                f"RETURN DISTINCT n.text AS label"
            )
            result = self.graph_adapter.execute_cypher_query(query)
            component = {node}
            component.update(row['label'] for row in result)
            components.append(component)
            visited.update(component)

        return components

    def _find_strongly_connected_components(self) -> List[Set[Any]]:
        """
        Find strongly connected components in the directed graph.

        Returns:
            List of sets, each containing node ids in a component.
        """
        # Fetch all nodes and edges
        nodes = self.get_nodes(with_data=False)
        edges = self.get_edges(with_data=False)

        # Build directed graph in Python
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        # Find strongly connected components
        sccs = list(nx.strongly_connected_components(g))
        return [set(comp) for comp in sccs]
