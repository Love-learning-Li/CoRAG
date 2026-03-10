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


from typing import Any, Dict, List, Tuple, Callable
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.utils.common import Lang

RAW_TEXT_KEY = 'raw_text'
FILE_ID_KEY = "file_id"
TEXT_CONCLUDE = "text_conclude"

CH_KEYS = {
    "entity": "实体",
    "event": "事件",
    "relation": "关系",
    "head_entity": "头实体",
    "tail_entity": "尾实体",
    "head_event": "头事件",
    "tail_event": "尾事件"
}
EN_KEYS = {
    "entity": "Entity",
    "event": "Event",
    "relation": "Relation",
    "head_entity": "Head",
    "tail_entity": "Tail",
    "head_event": "Head",
    "tail_event": "Tail"
}


ENTITY_TYPE = "entity"
EVENT_TYPE = "event"
NODE_TYPE = "type"


def extract_event_entity_triples(
    event_entity_relations: List[Dict[str, Any]], 
    keys: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """
    Extract (event, "participate", entity) triples from event-entity relation dicts.

    Args:
        event_entity_relations (List[Dict[str, Any]]): List of event-entity relation dicts.
        keys (Dict[str, str]): Key mapping dictionary.

    Returns:
        List[Tuple[str, str, str]]: List of (event, "participate", entity) triples.
    """
    triples = []
    for relation in event_entity_relations:
        if not isinstance(relation, dict):
            logger.warning("Wrong relation")
            continue
        event = relation.get(keys["event"])
        entities = relation.get(keys["entity"])
        if not isinstance(event, str) or not isinstance(entities, list):
            logger.warning("Invalid event-entity relation")
            continue
        for entity in entities:
            if isinstance(entity, str) and entity.strip():
                triples.append((event, "participate", entity.strip()))
    return triples


def get_language_keys(language: Lang) -> Dict[str, str]:
    """
    Retrieve the key mapping dictionary based on the language.

    Args:
        language (Lang): Language enum.

    Returns:
        Dict[str, str]: Key mapping dictionary.
    """
    return CH_KEYS if language == Lang.CH else EN_KEYS


def add_edge_with_attributes(
    graph: GraphStore,
    head: str,
    tail: str,
    relation: str,
    raw_text: str,
    file_id: str,
    head_type: str,
    tail_type: str
) -> None:
    """
    Add an edge and update node and edge attributes in the graph.

    Args:
        graph (GraphStore): The graph wrapper instance.
        head (str): Head node.
        tail (str): Tail node.
        relation (str): Relation type.
        raw_text (str): Raw text node.
        file_id (str): File identifier.
        head_type (str): Type of head node.
        tail_type (str): Type of tail node.
    """
    if head == tail:
        return
    graph.add_edge(head, tail, relation=relation)
    graph.add_edge(head, raw_text, relation=TEXT_CONCLUDE)
    graph.add_edge(tail, raw_text, relation=TEXT_CONCLUDE)

    graph.update_node_attribute(head, NODE_TYPE, head_type)
    graph.update_node_attribute(tail, NODE_TYPE, tail_type)

    graph.update_node_attribute(head, FILE_ID_KEY, file_id, append=True)
    graph.update_node_attribute(tail, FILE_ID_KEY, file_id, append=True)

    graph.update_edge_attribute(head, tail, FILE_ID_KEY, file_id, append=True)
    graph.update_edge_attribute(head, raw_text, FILE_ID_KEY, file_id, append=True)
    graph.update_edge_attribute(tail, raw_text, FILE_ID_KEY, file_id, append=True)


def process_entity_relations(
    graph: GraphStore,
    entity_relations: List[Dict[str, Any]],
    keys: Dict[str, str],
    raw_text: str,
    file_id: str
) -> List[str]:
    """
    Process and add entity relations to the graph, and return all unique relations.

    Args:
        graph (GraphStore): The graph wrapper instance.
        entity_relations (List[Dict[str, Any]]): List of entity relation dicts.
        keys (Dict[str, str]): Key mapping dictionary.
        raw_text (str): Raw text node.
        file_id (str): File identifier.

    Returns:
        List[str]: List of unique relation types.
    """
    for relation in entity_relations:
        if not isinstance(relation, dict):
            logger.warning("Wrong relation")
            continue
        head = relation.get(keys["head_entity"])
        rel = relation.get(keys["relation"])
        tail = relation.get(keys["tail_entity"])
        if all(isinstance(x, str) and x.strip() for x in (head, rel, tail)):
            add_edge_with_attributes(
                graph, head.strip(), tail.strip(), rel.strip(), raw_text, file_id, ENTITY_TYPE, ENTITY_TYPE
            )
        else:
            logger.warning("Invalid entity relation format")


def process_event_relations(
    graph: GraphStore,
    event_relations: List[Dict[str, Any]],
    keys: Dict[str, str],
    raw_text: str,
    file_id: str
) -> List[str]:
    """
    Process and add event relations to the graph, and return all unique relations.

    Args:
        graph (GraphStore): The graph wrapper instance.
        event_relations (List[Dict[str, Any]]): List of event relation dicts.
        keys (Dict[str, str]): Key mapping dictionary.
        raw_text (str): Raw text node.
        file_id (str): File identifier.

    Returns:
        List[str]: List of unique relation types.
    """
    for relation in event_relations:
        if isinstance(relation, list) and relation:
            relation = relation[0]
        if not isinstance(relation, dict):
            logger.warning("Wrong relation")
            continue
        head = relation.get(keys["head_event"])
        rel = relation.get(keys["relation"])
        tail = relation.get(keys["tail_event"])
        if all(isinstance(x, str) and x.strip() for x in (head, rel, tail)):
            add_edge_with_attributes(
                graph, head.strip(), tail.strip(), rel.strip(), raw_text, file_id, EVENT_TYPE, EVENT_TYPE
            )
        else:
            logger.warning("Invalid event relation format")


def process_event_entity_relations(
    graph: GraphStore,
    event_entity_relations: List[Dict[str, Any]],
    keys: Dict[str, str],
    raw_text: str,
    file_id: str
) -> List[str]:
    """
    Process and add event-entity participation relations to the graph, and return all unique relations.

    Args:
        graph (GraphStore): The graph wrapper instance.
        event_entity_relations (List[Dict[str, Any]]): List of event-entity relation dicts.
        keys (Dict[str, str]): Key mapping dictionary.
        raw_text (str): Raw text node.
        file_id (str): File identifier.

    Returns:
        List[str]: List of unique relation types.
    """
    triples = extract_event_entity_triples(event_entity_relations, keys)
    for event, relation, entity in triples:
        add_edge_with_attributes(
            graph, entity, event, relation, raw_text, file_id, ENTITY_TYPE, EVENT_TYPE
        )


def merge_relations_into_graph(
    graph: GraphStore,
    relations: List[Dict[str, Any]],
    language: Lang = Lang.CH
) -> None:
    """
    Efficiently merges a list of relation dicts into the graph.

    Args:
        graph (GraphStore): The graph wrapper instance.
        relations (List[Dict[str, Any]]): List of relation dicts.
        language (Lang, optional): Language enum. Defaults to Lang.CH.
    """
    keys = get_language_keys(language)

    for data in tqdm(relations, desc="Processing relations", total=len(relations)):
        if not isinstance(data, dict):
            logger.warning("Invalid relation")
            continue

        file_id = str(data.get(FILE_ID_KEY, ""))
        raw_text = data.get(RAW_TEXT_KEY, "")
        entity_relations = data.get("entity_relations", [])
        event_entity_relations = data.get("event_entity_relations", [])
        event_relations = data.get("event_relations", [])

        if not raw_text:
            logger.warning("Missing raw_text in relation")
            continue

        graph.add_node(raw_text)
        graph.update_node_attribute(raw_text, NODE_TYPE, RAW_TEXT_KEY)
        graph.update_node_attribute(raw_text, FILE_ID_KEY, file_id, append=True)

        process_entity_relations(graph, entity_relations, keys, raw_text, file_id)
        process_event_relations(graph, event_relations, keys, raw_text, file_id)
        process_event_entity_relations(graph, event_entity_relations, keys, raw_text, file_id)

    logger.info("Creating indices for graph...")
    if hasattr(graph, "create_index_for_edge"):
        graph.create_index_for_edge()
    if hasattr(graph, "create_index_for_node"):
        graph.create_index_for_node()

    logger.info(
        f"Graph stats - Nodes: {graph.number_of_nodes()}, "
        f"Edges: {graph.number_of_edges()}, Density: {graph.density():.6f}"
    )


class GraphMerger:
    """
    Merge relations into a graph and saving the result.

    Usage:
        merger = GraphMerger(graph)
        merger.merge(relations, language)
        merger.save(output_path)
    """

    def __init__(self, graph: GraphStore) -> None:
        """
        Initialize GraphMerger.

        Args:
            graph (GraphStore): The graph instance.
        """
        self.graph = graph

    def merge(self, relations: List[Dict[str, Any]], language: Lang = Lang.CH) -> "GraphMerger":
        """
        Merge relations into the graph.

        Args:
            relations: List of relation dicts.
            language: Language enum. Defaults to `Lang.CH`.

        Returns:
            GraphMerger: Returns self for method chaining.
        """
        merge_relations_into_graph(self.graph, relations, language)

    def save_graph(self, output_path: str, encrypt_fn: Callable) -> None:
        """
        Save the graph to the specified output path.

        Args:
            encrypt_fn: function to encrypt text
            output_path (str): Path to save the graph.
        """
        self.graph.save(output_path, encrypt_fn)
