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

import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langchain_opengauss import openGaussAGEGraph, OpenGaussSettings

from mx_rag.utils.common import MAX_RECURSION_LIMIT



def cypher_value(v, depth=0, seen=None):
    """
    Convert a Python value to a safe representation for Cypher queries

    Parameters:
        v: The value to convert (str, int, float, bool, None, list, dict)
        depth: The current recursion depth (default is 0)
        seen: A set of object IDs already processed to detect circular references (default is None)

    Returns:
        str: A string representation safe to embed in Cypher queries

    Raises:
        ValueError: If the structure is too deep or contains circular references
    """
    if seen is None:
        seen = set()

    if depth > MAX_RECURSION_LIMIT:
        raise ValueError("Structure too deep - possible circular reference")

    if id(v) in seen:
        raise ValueError("Circular reference detected")
    seen.add(id(v))

    try:
        if v is None:
            return 'null'
        elif isinstance(v, bool):
            return str(v).lower()
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, str):
            # Escape single quotes, backslashes, and special Cypher characters for string literals
            # This prevents injection attacks by escaping characters that could break out of strings
            escaped = v.replace("\\", "\\\\").replace("'", "\\'")
            # Escape parentheses to prevent Cypher command injection
            escaped = escaped.replace("(", "\\\\(").replace(")", "\\\\)")
            return f"'{escaped}'"
        elif isinstance(v, (list, tuple, dict)):
            new_seen = set(seen)
            if isinstance(v, (list, tuple)):
                items = [cypher_value(item, depth + 1, new_seen) for item in v]
                return f'[{", ".join(items)}]'
            else:
                pairs = [f'{key}: {cypher_value(value, depth + 1, new_seen)}' for key, value in v.items()]
                return f'{{{", ".join(pairs)}}}'
        else:
            raise ValueError(f"Unsupported type for Cypher value: {type(v)}")
    finally:
        seen.remove(id(v))


def escape_identifier(identifier: str) -> str:
    """
    Validate and escape an identifier for safe use in Cypher.

    Identifiers must only contain alphanumeric characters, underscores, and hyphens.
    This prevents injection attacks through identifiers.

    Parameters:
        identifier: The identifier to validate and escape

    Returns:
        str: The validated identifier

    Raises:
        ValueError: If the identifier is invalid or contains unsafe characters
    """
    if not isinstance(identifier, str):
        raise ValueError("Identifier must be a string")

    if not identifier:
        raise ValueError("Identifier cannot be empty")

    # Allow only alphanumeric characters, underscores, and hyphens
    # This prevents Cypher injection through identifiers
    if not all(c.isalnum() or c in ('_', '-') for c in identifier):
        raise ValueError(f"Invalid identifier: {identifier}. Only alphanumeric, underscore, and hyphen allowed.")

    # Identifiers cannot start with a number in Cypher
    if identifier[0].isdigit():
        raise ValueError(f"Invalid identifier: {identifier}. Cannot start with a digit.")

    return identifier


class OpenGaussAGEAdapter:
    """
    Adapter class that extends openGaussAGEGraph to expose additional utility methods
    for database operations while maintaining full compatibility with the parent class.
    """

    def __init__(self, age_graph: openGaussAGEGraph):
        """
        Initialize the adapter by calling the parent constructor.
        Args:
            age_graph: openGaussAGEGraph instance
        """
        self.age_graph = age_graph

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @contextmanager
    def get_cursor(self):
        """
        Expose the _get_cursor method as a public method.

        Returns:
            A database cursor context manager
        """
        cursor = self.age_graph.connection.cursor()
        yield cursor

    def execute_cypher_query(self, cypher_query: str) -> Any:
        """
        Execute a Cypher query through the graph instance.

        Args:
            cypher_query (str): The Cypher query to execute

        Returns:
            Query results
        """
        return self.age_graph.query(cypher_query)

    def close(self):
        """Close the database connection."""
        if hasattr(self.age_graph, 'connection') and self.age_graph.connection:
            self.age_graph.connection.close()

    @property
    def connection(self):
        return self.age_graph.connection


class CypherQueryBuilder:
    """Helper class for building Cypher queries."""

    @staticmethod
    def merge_node(attributes: Dict[str, Any]) -> str:
        query = f"CREATE (n:Node {cypher_value(attributes)})"
        return query

    @staticmethod
    def match_node(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}}) RETURN n LIMIT 1"

    @staticmethod
    def delete_node(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}}) DETACH DELETE n"

    @staticmethod
    def match_node_properties(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}}) RETURN properties(n) AS props"

    @staticmethod
    def match_node_attribute(label: str, key: str) -> str:
        safe_key = escape_identifier(key)
        return f"MATCH (n:Node {{id: {cypher_value(label)}}}) RETURN n.{safe_key} AS value"

    @staticmethod
    def set_node_attribute(label: str, key: str, value, append: bool = False) -> str:
        safe_key = escape_identifier(key)
        val = cypher_value(value)
        if append:
            return (
                f"MATCH (n:Node {{id: {cypher_value(label)}}}) "
                f"WITH n, CASE WHEN coalesce(n.{safe_key}, '') = '' THEN {val} else n.{safe_key} + ',' + {val} END AS new_value "
                f"SET n.{safe_key} = new_value"
            )
        return f"MATCH (n:Node {{id: {cypher_value(label)}}}) SET n.{safe_key} = {cypher_value(value)}"

    @staticmethod
    def set_node_attributes(name: str, props) -> str:
        safe_name = escape_identifier(name)
        return (
            f"UNWIND {cypher_value(props)} AS item "
            f"MATCH (n:Node) WHERE n.id = item.label "
            f"SET n.{safe_name} = item.value"
        )

    @staticmethod
    def match_nodes(with_data: bool = True) -> str:
        if with_data:
            return "MATCH (n:Node) RETURN n.text AS label, properties(n) AS props"
        return "MATCH (n:Node) RETURN n.text AS label"

    @staticmethod
    def match_nodes_by_attribute(key: str, value) -> str:
        safe_key = escape_identifier(key)
        return f"MATCH (n:Node) WHERE n.{safe_key} = {cypher_value(value)} RETURN properties(n) AS props"

    @staticmethod
    def match_nodes_containing_attribute(key: str, value: str) -> str:
        safe_key = escape_identifier(key)
        return (
            f"MATCH (n:Node) WHERE toString(n.{safe_key}) CONTAINS {cypher_value(value)} "
            "RETURN properties(n) AS props"
        )

    @staticmethod
    def merge_edge(source_label: str, target_label: str, attributes: Dict[str, Any]) -> str:
        props = cypher_value(attributes)
        relation = cypher_value(attributes.get("relation", "related"))
        if props:
            query = (
                f"MATCH (a:Node {{id: {cypher_value(source_label)}}}), (b:Node {{id: {cypher_value(target_label)}}}) "
                f"MERGE (a)-[r:`{relation}` {props}]->(b)"
            )
        else:
            query = (
                f"MATCH (a:Node {{id: {cypher_value(source_label)}}}), (b:Node {{id: {cypher_value(target_label)}}}) "
                f"MERGE (a)-[r:`{relation}`]->(b)"
            )
        return query

    @staticmethod
    def delete_edge(source_label: str, target_label: str) -> str:
        return f"MATCH (a:Node {{id: {cypher_value(source_label)}}})-[r]->(b:Node {{id: {cypher_value(target_label)}}}) DELETE r"

    @staticmethod
    def match_edge(source_label: str, target_label: str) -> str:
        return f"MATCH (a:Node {{id: {cypher_value(source_label)}}})-[r]->(b:Node {{id: {cypher_value(target_label)}}}) RETURN r LIMIT 1"

    @staticmethod
    def match_edges(with_data: bool = True) -> str:
        base = (
            "MATCH (a:Node)-[r]->(b:Node) "
            "RETURN a.text AS source, b.text AS target, a.id AS start_id, b.id AS end_id"
        )
        if with_data:
            return f"{base}, properties(r) AS props"
        return base

    @staticmethod
    def match_edge_attribute(source_label: str, target_label: str, key: Optional[str] = None) -> str:
        if key:
            safe_key = escape_identifier(key)
            return (
                f"MATCH (:Node {{id: {cypher_value(source_label)}}})-[r]->(:Node {{id: {cypher_value(target_label)}}}) "
                f"RETURN r.{safe_key} AS value"
            )
        return (
            f"MATCH (:Node {{id: {cypher_value(source_label)}}})-[r]->(:Node {{id: {cypher_value(target_label)}}}) "
            f"RETURN properties(r) AS props"
        )

    @staticmethod
    def set_edge_attribute(source_label: str, target_label: str, key: str, value, append: bool = False) -> str:
        safe_key = escape_identifier(key)
        if append:
            return (
                f"MATCH (a:Node {{id: {cypher_value(source_label)}}})-[r]->(b:Node {{id: {cypher_value(target_label)}}}) "
                f"SET r.{safe_key} = coalesce(r.{safe_key}, []) + {cypher_value(value)}"
            )
        return (
            f"MATCH (a:Node {{id: {cypher_value(source_label)}}})-[r]->(b:Node {{id: {cypher_value(target_label)}}}) "
            f"SET r.{safe_key} = {cypher_value(value)}"
        )

    @staticmethod
    def match_edges_by_attribute(key: str) -> str:
        safe_key = escape_identifier(key)
        return (
            f"MATCH (a:Node)-[r]->(b:Node) WHERE exists(r.{safe_key}) "
            "RETURN a.id as source, b.id as target, properties(r) AS props"
        )

    @staticmethod
    def in_degree(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}})<-[r]-() RETURN count(r) AS deg"

    @staticmethod
    def out_degree(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}})-[r]->() RETURN count(r) AS deg"

    @staticmethod
    def neighbors(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}})--(m) RETURN m.text as label"

    @staticmethod
    def successors(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}})-->(m) RETURN m.text as label"

    @staticmethod
    def predecessors(label: str) -> str:
        return f"MATCH (n:Node {{id: {cypher_value(label)}}})<--(m) RETURN m.text as label"

    @staticmethod
    def count_nodes() -> str:
        return "MATCH (n:Node) RETURN count(n) AS cnt"

    @staticmethod
    def count_edges() -> str:
        return "MATCH ()-[r]->() RETURN count(r) AS cnt"
