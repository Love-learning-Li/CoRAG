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

import random
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import Lang, MAX_PROMPT_LENGTH, validate_params
from mx_rag.graphrag.prompts.extract_graph import (
    ENTITY_PROMPT_CN,
    ENTITY_PROMPT_EN,
    EVENT_PROMPT_CN,
    EVENT_PROMPT_EN,
    RELATION_PROMPT_CN,
    RELATION_PROMPT_EN,
)
from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.graphrag.graphs.opengauss_graph import OpenGaussGraph


def extract_event_nodes(graph) -> List[Any]:
    """
    Extract all event nodes from the graph.
    """
    return graph.get_nodes_by_attribute(key="type", value="event")


def extract_entity_nodes(graph) -> List[Any]:
    """
    Extract all entity nodes from the graph.
    """
    return graph.get_nodes_by_attribute(key="type", value="entity")


def extract_relation_edges(graph) -> List[Any]:
    """
    Extract all relation edges from the graph.
    """
    seen = set()
    relations = []
    for relation in graph.get_edge_attribute_values(key="relation"):
        if relation not in seen:
            seen.add(relation)
            relations.append(relation)
    return relations


def _check_conceptualizer_prompts(prompts: Optional[dict]) -> bool:
    """
    Check if the conceptualizer prompts are valid.
    """
    if prompts is None:
        return True

    required_keys = {"event", "entity", "relation"}
    if not isinstance(prompts, dict) or set(prompts.keys()) != required_keys:
        return False
    return all(isinstance(prompts[key], str) and 0 < len(prompts[key]) <= MAX_PROMPT_LENGTH for key in required_keys)


@validate_params(
    llm=dict(
        validator=lambda x: isinstance(x, Text2TextLLM),
        message="llm must be an instance of Text2TextLLM",
    ),
    graph=dict(
        validator=lambda x: isinstance(x, GraphStore),
        message="graph must be an instance of GraphStore",
    ),
    sample_num=dict(
        validator=lambda x: x is None or (isinstance(x, int) and 0 <= x <= 1000000),
        message="sample_num must be None or an integer, value range [0, 1000000]",
    ),
    lang=dict(
        validator=lambda x: isinstance(x, Lang),
        message="lang must be an instance of Lang",
    ),
    seed=dict(
        validator=lambda x: isinstance(x, int) and 0 <= x <= 65535,
        message="seed must be an integer, value range [0, 65535]",
    ),
    prompts=dict(
        validator=lambda x: _check_conceptualizer_prompts(x),
        message="prompts must be None or a dict with keys: 'event', 'entity', 'relation'",
    ),
)
class GraphConceptualizer:
    """
    Conceptualizes events, entities, and relations in a graph using an LLM.
    """

    def __init__(
            self,
            llm: Text2TextLLM,
            graph: GraphStore,
            sample_num: Optional[int] = None,
            lang: Lang = Lang.CH,
            seed: int = 4096,
            prompts: Optional[dict] = None,
            max_workers=None
    ) -> None:
        random.seed(seed)
        self.llm = llm
        self.graph = graph
        self.sample_num = sample_num
        self.max_workers = max_workers
        self.prompts = prompts or {
            "event": (EVENT_PROMPT_CN if lang == Lang.CH else EVENT_PROMPT_EN),
            "entity": (ENTITY_PROMPT_CN if lang == Lang.CH else ENTITY_PROMPT_EN),
            "relation": (RELATION_PROMPT_CN if lang == Lang.CH else RELATION_PROMPT_EN),
        }

        self.events = extract_event_nodes(self.graph)
        self.entities = extract_entity_nodes(self.graph)
        self.relations = extract_relation_edges(self.graph)

        if sample_num:
            self.events = random.sample(self.events, min(sample_num, len(self.events)))
            self.entities = random.sample(
                self.entities, min(sample_num, len(self.entities))
            )
            self.relations = random.sample(
                self.relations, min(sample_num, len(self.relations))
            )

    def conceptualize(self) -> List[Dict[str, Any]]:
        """
        Conceptualize events, entities, and relations in the graph in parallel.

        Returns:
            List of conceptualized nodes and relations.
        """
        result = []

        def run_parallel(items, func, desc):
            outputs = []
            with ThreadPoolExecutor(self.max_workers) as executor:
                future_to_item = {executor.submit(func, item): item for item in items}
                for future in tqdm(
                        as_completed(future_to_item), total=len(items), desc=desc
                ):
                    outputs.append(future.result())
            return outputs

        result.extend(
            run_parallel(
                self.events, self._conceptualize_event, "Conceptualizing events"
            )
        )
        result.extend(
            run_parallel(
                self.entities, self._conceptualize_entity, "Conceptualizing entities"
            )
        )
        result.extend(
            run_parallel(
                self.relations,
                self._conceptualize_relation,
                "Conceptualizing relations",
            )
        )

        return result

    def _conceptualize_event(self, event: str) -> Dict[str, Any]:
        """
        Conceptualize a single event node.

        Args:
            event: The event node.

        Returns:
            Dict with conceptualized event.
        """
        prompt = self.prompts["event"].replace("[EVENT]", event)
        answer = self.llm.chat(prompt)
        return {
            "node": event,
            "conceptualized_node": answer,
            "node_type": "event",
        }

    def _conceptualize_entity(self, entity: str) -> Dict[str, Any]:
        """
        Conceptualize a single entity node.

        Args:
            entity: The entity node.

        Returns:
            Dict with conceptualized entity.
        """
        entity_name = entity.split(":::")[0] if ":::" in entity else entity
        prompt = self.prompts["entity"].replace("[ENTITY]", entity_name)

        if isinstance(self.graph, OpenGaussGraph):
            # Multi-thread case: each thread gets its connection
            local_graph = OpenGaussGraph(self.graph.graph_name, self.graph.age_graph)
        else:
            local_graph = self.graph
        entity_predecessors = list(local_graph.predecessors(entity))
        entity_successors = list(local_graph.successors(entity))

        context = ""
        if entity_predecessors:
            neighbors = random.sample(
                entity_predecessors, min(1, len(entity_predecessors))
            )
            context += ", ".join(
                f"{neighbor} {local_graph.get_edge_attributes(neighbor, entity, 'relation')}"
                for neighbor in neighbors
            )
        if entity_successors:
            neighbors = random.sample(entity_successors, min(1, len(entity_successors)))
            if context:
                context += ", "
            context += ", ".join(
                f"{local_graph.get_edge_attributes(entity, neighbor, 'relation')} {neighbor}"
                for neighbor in neighbors
            )

        prompt = prompt.replace("[CONTEXT]", context)
        answer = self.llm.chat(prompt)
        return {
            "node": entity,
            "conceptualized_node": answer,
            "node_type": "entity",
        }

    def _conceptualize_relation(self, relation: str) -> Dict[str, Any]:
        """
        Conceptualize a single relation.

        Args:
            relation: The relation.

        Returns:
            Dict with conceptualized relation.
        """
        prompt = self.prompts["relation"].replace("[RELATION]", relation)
        answer = self.llm.chat(prompt)
        return {
            "node": relation,
            "conceptualized_node": answer,
            "node_type": "relation",
        }
