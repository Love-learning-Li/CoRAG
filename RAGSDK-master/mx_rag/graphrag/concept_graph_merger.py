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


from collections import Counter
from typing import Any, Dict, List, Callable, Set
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.graphs.graph_store import GraphStore


class ConceptGraphMerger:
    """
    Merges conceptual and synset information into a graph database, updating node and edge attributes accordingly.
    """

    def __init__(self, graph: GraphStore, batch_size: int = 4) -> None:
        """
        Initializes the ConceptGraphMerger with a graph database object.
        """
        self.graph = graph
        self.entity_concepts: Dict[str, List[str]] = {}
        self.relation_concepts: Dict[str, List[str]] = {}
        self.event_concepts: Dict[str, List[str]] = {}
        self.concept_counter: Counter = Counter()
        self.entity_concept_counter: Counter = Counter()
        self.relation_concept_counter: Counter = Counter()
        self.event_concept_counter: Counter = Counter()
        self.node_type_counter: Dict[str, int] = {"entity": 0, "relation": 0, "event": 0}
        self.concept_to_synset: Dict[str, List[str]] = {}
        self.synset_counter: Counter = Counter()
        self.batch_size = batch_size

    @staticmethod
    def parse_concept_string(concepts_str: str) -> List[str]:
        return [c for c in dict.fromkeys(map(str.strip, concepts_str.split(","))) if c]

    def merge_concepts_and_synset(
            self, concept_data: List[Dict[str, Any]], synset_list: List[Set[str]]
    ) -> None:
        self._process_concept_data(concept_data)
        self._process_synset(synset_list)
        self._update_graph_attributes()

    def save_graph(self, output_path: str, encrypt_fn: Callable):
        """
        Saves the graph database to the specified output path.
        """
        self.graph.save(output_path, encrypt_fn)

    def _process_concept_data(self, concept_data: List[Dict[str, Any]]) -> None:
        """
        Processes concept data and updates internal counters and mappings.
        """
        for concept in concept_data:
            node_text = concept["node"]
            concepts_str = concept.get("conceptualized_node", "")
            concepts = self.parse_concept_string(concepts_str)
            node_type = concept["node_type"]
            self.node_type_counter[node_type] += 1
            self._collect_node_concepts(node_type, node_text, concepts)
        logger.info(f"Node type counts: {self.node_type_counter}")
        logger.info("Raw concepts processed.")

    def _collect_node_concepts(self, node_type: str, node_text: str, concepts: List[str]) -> None:
        """
        Collects and counts concepts for a given node type and text.
        """
        if node_type == "entity":
            self._update_concept_mappings(self.entity_concepts, self.entity_concept_counter, node_text, concepts)
        elif node_type == "relation":
            self._update_concept_mappings(self.relation_concepts, self.relation_concept_counter, node_text, concepts)
        elif node_type == "event":
            self._update_concept_mappings(self.event_concepts, self.event_concept_counter, node_text, concepts)
        else:
            logger.warning(f"Unknown node type: {node_type}")

    def _update_concept_mappings(
            self,
            concept_dict: Dict[str, List[str]],
            concept_counter: Counter,
            node_text: str,
            concepts: List[str]
    ) -> None:
        """
        Updates concept mappings and counters for a node.
        """
        concept_dict[node_text] = concepts
        for concept in concepts:
            concept_counter[concept] += 1
            self.concept_counter[concept] += 1

    def _process_synset(self, synset_list: List[List[str]]) -> None:
        """
        Processes synset and updates concept-to-synset mapping and synset counters.
        """
        logger.info("Processing synset...")
        for synset in synset_list:
            concept_with_freq = [[concept, self.concept_counter[concept]] for concept in synset]
            sorted_concept_with_freq = sorted(concept_with_freq, key=lambda x: x[1], reverse=True)
            sorted_concepts = [concept for concept, _ in sorted_concept_with_freq]
            for concept in sorted_concepts:
                self.concept_to_synset[concept] = sorted_concepts

        for synset in tqdm(self.concept_to_synset.values(),
                           total=len(self.concept_to_synset.values()), desc="Counting synset"):
            synset_tuple = tuple(synset)
            self.synset_counter[synset_tuple] += sum(self.concept_counter[c] for c in synset)

    def _update_graph_attributes(self) -> None:
        """
        Updates the graph database with concept and synset attributes for nodes and edges using batch updates.
        """
        # Use sets to track processed nodes and avoid duplicates
        processed_nodes = set()
        node_updates = []
        edge_updates = []

        graph_edges = self.graph.get_edges()

        # Process each edge once: O(E)
        for u, v, data in tqdm(graph_edges, desc="Preparing graph attributes"):
            # Process both nodes using the helper method
            self._process_node_if_not_seen(u, processed_nodes, node_updates)
            self._process_node_if_not_seen(v, processed_nodes, node_updates)

            # Process edge relation concepts
            concepts_r = self.relation_concepts.get(data.get("relation", ""), [])
            synset_r = self._get_synset_strings(concepts_r)

            concept_r_string = ",".join(concepts_r) if concepts_r else ""
            synset_r_string = ",".join(synset_r) if synset_r else ""

            edge_updates.append((u, v, {"concepts": concept_r_string, "synset": synset_r_string}))

        # Batch update nodes and edges
        self.graph.update_node_attributes_batch(node_updates, self.batch_size)
        self.graph.update_edge_attributes_batch(edge_updates, self.batch_size)

    def _process_node_if_not_seen(self, node: str, processed_nodes: set, node_updates: list) -> None:
        """
        Processes a node for concepts and synsets if it hasn't been processed yet.
        """
        if node not in processed_nodes:
            node_type = self.graph.get_node_attributes(node, "type")
            concepts = self._get_concepts_by_type(node, node_type)
            synset = self._get_synset_strings(concepts)

            concept_string = ",".join(concepts) if concepts else ""
            synset_string = ",".join(synset) if synset else ""

            node_updates.append((node, {"concepts": concept_string, "synset": synset_string}))
            processed_nodes.add(node)

    def _get_concepts_by_type(self, node: str, node_type: str) -> List[str]:
        """
        Retrieves concepts for a node based on its type.
        """
        if node_type is None:
            return []

        if "entity" in node_type:
            return self.entity_concepts.get(node, [])
        elif node_type == "event":
            return self.event_concepts.get(node, [])
        return []

    def _get_synset_strings(self, concepts: List[str]) -> List[str]:
        """
        Retrieves the primary synset string for each concept, removing empty and duplicate synset.
        """
        synset_strings = set()
        for concept in concepts:
            synset = self.concept_to_synset.get(concept, [])
            if synset:
                synset_strings.add(synset[0])
        return list(synset_strings)
