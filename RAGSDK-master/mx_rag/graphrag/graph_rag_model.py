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

import concurrent.futures
import time
from collections import Counter
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.graphs.graph_store import GraphStore
from mx_rag.graphrag.graphs.opengauss_graph import OpenGaussGraph
from mx_rag.graphrag.prompts.evaluate_qa import TEXT_RAG_TEMPLATE
from mx_rag.graphrag.qa_base_model import QABaseModel
from mx_rag.graphrag.vector_stores.vector_store_wrapper import VectorStoreWrapper
from mx_rag.llm import LLMParameterConfig, Text2TextLLM
from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import validate_params


class GraphRAGModel(QABaseModel):
    """
    GraphRAGModel integrates graph-based retrieval-augmented generation with LLMs.
    It manages embedding databases, retrieves relevant nodes, and generates answers using graph context.
    """

    @validate_params(
        reranker_top_k=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= 1000, 
            message="param must be an integer, value range [1, 1000]"
        ),
        retrieval_top_k=dict(
            validator=lambda x: isinstance(x, int) and 0 < x <= 1000,
            message="param must be an integer, value range [1, 1000]"
        ),
        subgraph_depth=dict(
            validator=lambda x: isinstance(x, int) and 1 <= x < 6, 
            message="param must be an integer, value range [1, 5]"
        ),
        similarity_tail_threshold=dict(
            validator=lambda x: isinstance(x, (float, int)) and 0.0 <= x <= 1.0, 
            message="param must be float or int and value range [0.0, 1.0]"
        ),
        use_text=dict(validator=lambda x: isinstance(x, bool), message="param must be a boolean")
    )
    def __init__(
        self,
        llm: Text2TextLLM,
        llm_config: LLMParameterConfig,
        embed_func: Callable[[List[str], int], List[Any]],
        graph_store: GraphStore,
        vector_store: VectorStoreWrapper,
        metric: str = "generation",
        vector_store_concept: Optional[VectorStoreWrapper] = None,
        reranker: Optional[Reranker] = None,
        retrieval_top_k: int = 10,
        reranker_top_k: int = 10,
        subgraph_depth: int = 1,
        use_text: bool = False,
        batch_size=4,
        similarity_tail_threshold=0.3,
        min_number_texts=3
    ):
        """
        Initialize the GraphRAGModel with required components and configuration.
        """
        super().__init__(llm, llm_config, metric)
        self.embed_func = embed_func
        self.graph = graph_store
        self.vector_store = vector_store
        self.vector_store_concept = vector_store_concept
        self.reranker = reranker
        self.subgraph = None
        self.retrieval_top_k = retrieval_top_k
        self.reranker_top_k = reranker_top_k
        self.subgraph_depth = subgraph_depth
        self.use_text = use_text
        self.batch_size = batch_size
        self.similarity_tail_threshold = similarity_tail_threshold
        self.min_number_text = min_number_texts
        self.node_names: List[str] = []
        self.text_nodes: List[str] = []
        self.concepts: List[str] = []
        self._initialize_databases()

    @staticmethod
    def _gather_nodes_for_question(entities: List[str], entity_to_nodes: dict) -> List[str]:
        """Gather and deduplicate nodes for a single question."""
        retrieved_nodes = []
        seen_nodes = set()

        for entity in entities:
            for node in entity_to_nodes.get(entity, []):
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    retrieved_nodes.append(node)

        return retrieved_nodes

    def _safe_embed_func(self, *args, **kwargs):
        embeddings = self.embed_func(*args, **kwargs)
        if not (isinstance(embeddings, (List, np.ndarray)) and len(embeddings) > 0):
            raise ValueError(f"callback function {self.embed_func.__name__}"
                             f" returned invalid result, should be List[Any]")
        return embeddings

    def search_index(self, query, top_k) -> List[str]:
        try:

            query_embedding = np.asarray(self._safe_embed_func([query]))
            _, idx = self.vector_store.search(query_embedding, top_k)
            idx = idx[0] if idx is not None and len(idx) > 0 else []
            
            text_nodes_set = set(self.text_nodes)
            retrieved = [self.node_names[i] for i in idx if self.node_names[i] in text_nodes_set]
            return retrieved
        except TypeError as e:
            logger.error(f"Type error in search_index: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error in search_index: {e}")
            raise
        except Exception as e:
            logger.error(f"search_index error: {e}")
            raise
    
    @validate_params(top_k=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 1000,
                                message="top_k must be an integer, value range in [1, 1000]"))
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves top-k relevant node names for a given query using node and concept embeddings.

        Args:
            query: The input query string.
            top_k: Number of top nodes to retrieve.

        Returns:
            List of retrieved node names.
        """
        try:
            query_embedding = np.asarray(self._safe_embed_func([query]))
            _, idx = self.vector_store.search(query_embedding, top_k)
            retrieved = [self.node_names[i] for i in idx[0] if i != -1] if idx and len(idx[0]) > 0 else []

            if self.vector_store_concept is not None:
                _, idx_concept = self.vector_store_concept.search(query_embedding, top_k)
                concept_nodes = (
                    [self.node_names[i] for i in idx_concept[0] if i != -1] 
                    if idx_concept and len(idx_concept[0]) > 0 else []
                )
                # Merge and deduplicate, preserving order and prioritizing most frequent
                all_nodes = retrieved + concept_nodes
                return [item for item, _ in Counter(all_nodes).most_common(top_k)]

            return retrieved
        except TypeError as e:
            logger.error(f"Type error in retrieve: {e}")
            return []
        except ValueError as e:
            logger.error(f"Value error in retrieve: {e}")
            return []
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @validate_params(nodes=dict(validator=lambda x: isinstance(x, list) and len(x) < 100000,
                                message="nodes must be a list and its length less than 100000"),
                     n=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 5,
                            message="n must be an integer between 1 and 5"))
    def get_contexts_for_nodes(self, nodes: List[str], n: int) -> List[str]:
        """
        Extracts contexts for the given nodes up to n-order neighbors.

        Args:
            nodes: List of node names to extract contexts from.
            n: The neighbor depth (order) to traverse for subgraph extraction.

        Returns:
            contexts: List of contexts.
        """
        if isinstance(self.graph, OpenGaussGraph):
            triples = self.graph.subgraph(nodes, n)
        else:
            self._build_neighbor_subgraph(nodes, n)
            triples = self._extract_edges_with_attributes()
        if not self.use_text:
            return [f"{u} {r} {v}" for u, r, v in triples]

        text_nodes = []
        seen = set()
        for _, r, v in triples:
            if r == 'text_conclude' and v not in seen:
                seen.add(v)
                text_nodes.append(v)

        if not text_nodes:
            return []
        
        return text_nodes
    
    def reset_subgraph(self) -> None:
        """
        Resets the current subgraph.
        """
        del self.subgraph
        self.subgraph = None

    @validate_params(questions=dict(validator=lambda x: isinstance(x, list) and len(x) < 10000,
                                    message="questions must be a list and its length less than 10000"))
    def generate(self, questions: List[str], max_triples: int = 150, retrieve_only: bool = True) -> List[str]:
        """
        Generates answers for a list of questions using graph-based retrieval and LLM.

        Args:
            questions: List of question strings.
            max_triples: Maximum number of triples to include in the prompt.

        Returns:
            List of generated responses.
        """
        logger.info("Generating using graph...")
        
        # Step 1: Extract entities from all questions
        entities_list = self._extract_entities_batch(questions)
        
        # Step 2: Retrieve nodes for all unique entities
        entity_to_nodes = self._retrieve_nodes_batch(entities_list)
        
        # Step 3: Prepare prompts for all questions
        prompts, all_contexts = self._prepare_prompts_batch(questions, entities_list, entity_to_nodes, max_triples)
        
        # Step 4: Generate answers in parallel
        return all_contexts if retrieve_only else self._generate_answers_batch(prompts)

    def _extract_entities_batch(self, questions: List[str]) -> List[List[str]]:
        """Extract entities from all questions in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(tqdm(
                executor.map(self._extract_entities_from_question, questions),
                total=len(questions),
                desc="Extracting entities"
            ))

    def _retrieve_nodes_batch(self, entities_list: List[List[str]]) -> dict:
        """Retrieve nodes for all unique entities."""
        all_entities = set(entity for entities in entities_list for entity in entities)
        
        def retrieve_entity(entity):
            return entity, self.retrieve(entity, top_k=self.retrieval_top_k)
        
        return dict(tqdm(
            map(retrieve_entity, all_entities),
            total=len(all_entities),
            desc="Retrieving nodes"
        ))

    def _prepare_prompts_batch(
        self, 
        questions: List[str], 
        entities_list: List[List[str]], 
        entity_to_nodes: dict, 
        max_triples: int
    ) -> Tuple[List[str], List[List[str]]]:
        """Prepare prompts for all questions."""
        prompts = []
        all_contexts = []
        
        for question, entities in tqdm(
            zip(questions, entities_list), 
            total=len(questions), 
            desc="Preparing prompts"
        ):
            # Gather and deduplicate nodes for current question
            retrieved_nodes = self._gather_nodes_for_question(entities, entity_to_nodes)
            
            # Get and rerank contexts
            contexts = self._get_and_rerank_contexts(retrieved_nodes, question, max_triples)
            all_contexts.append(contexts)
            
            # Create prompt
            prompt = TEXT_RAG_TEMPLATE.format(context=contexts, question=question)
            prompts.append(prompt)
        
        return prompts, all_contexts

    def _get_and_rerank_contexts(
        self, 
        retrieved_nodes: List[str], 
        question: str, 
        max_triples: int
    ) -> List[str]:
        """Get graph contexts and rerank them."""
        logger.debug(f"Retrieved nodes count: {len(retrieved_nodes)}")
        
        # Get graph contexts with timing
        start_time = time.time()
        contexts = self.get_contexts_for_nodes(retrieved_nodes, self.subgraph_depth)[:max_triples]
        context_time = (time.time() - start_time) * 1000
        
        if context_time > 100:  # Only log timing if it's slow(>100ms)
            logger.debug(f"Context retrieval: {context_time:.4f}ms for {len(contexts)} contexts")
        
        # Rerank contexts
        return self._rerank(contexts, question) if contexts else []

    def _generate_answers_batch(self, prompts: List[str]) -> List[str]:
        """Generate answers for all prompts in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(tqdm(
                executor.map(self._call_llm_with_retry, prompts),
                total=len(prompts),
                desc="Generating answers (parallel)"
            ))

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic."""
        for attempt in range(1, max_retries + 1):
            response = self.llm.chat(prompt, llm_config=self.llm_config)
            if response.strip():
                return response
            logger.warning(f"Failed to get response, retry {attempt}")
        
        logger.warning(f'No response from LLM after {max_retries} attempts.')
        return ""

    def _initialize_databases(self) -> None:
        self._build_node_database()
        if self.vector_store_concept is not None:
            self._build_concept_database()
        logger.info("Databases initialized.")

    def _build_node_database(self) -> None:
        """
        Builds or updates the node embedding database efficiently.
        """
        # Get all unique, non-empty node names as strings
        self.node_names = [str(node) for node in self.graph.get_nodes(with_data=False) if str(node).strip()]
        for node, data in self.graph.get_nodes():
            if str(node).strip() and data["type"] == "raw_text":
                self.text_nodes.append(str(node))
        node_count = len(self.node_names)
        # Only rebuild if counts mismatch
        if self.vector_store.ntotal() != node_count:
            logger.info("Building node embedding database...")
            # Batch embedding for efficiency
            embeddings = self._safe_embed_func(self.node_names, batch_size=self.batch_size)
            # Efficiently clear and add to vector store
            self.vector_store.clear()
            self.vector_store.add(np.asarray(embeddings), np.arange(node_count).tolist())
            self.vector_store.save()

    def _build_concept_database(self) -> None:
        """
        Builds or updates the concept embedding database.
        """
        # Collects all unique, non-empty concepts from node attributes
        concepts_set = set()
        for _, data in self.graph.get_nodes():
            concept = data.get("concepts")
            if concept:
                if isinstance(concept, (list, set, tuple)):
                    concepts_set.update(map(str, concept))
                else:
                    concepts_set.add(str(concept))
        self.concepts = list(concepts_set)

        if self.vector_store_concept and self.vector_store_concept.ntotal() != len(self.concepts):
            logger.info("Building concept embedding database...")
            embeddings = self._safe_embed_func(self.concepts, batch_size=self.batch_size)
            self.vector_store_concept.clear()
            self.vector_store_concept.add(np.array(embeddings), list(range(len(embeddings))))
            self.vector_store_concept.save()

    def _rerank(self, contexts, query):
        if self.use_text:
            if self.reranker is None:
                return contexts
            else:
                scores = self.reranker.rerank(query, contexts)
                items = self.reranker.rerank_top_k(contexts, scores)
        else:
            items = [item for item, _ in Counter(contexts).most_common(self.reranker_top_k)]
        return items

    def _add_neighbors_to_subgraph(
        self,
        current_node: Any,
        visited: Set[Any],
        queue: List[Tuple[Any, int]],
        current_distance: int
    ) -> None:
        """
        Adds neighbors and predecessors of the current node to the subgraph and queue.

        Args:
            current_node: The node to expand.
            visited: Set of already visited nodes.
            queue: Queue for BFS traversal.
            current_distance: Current BFS depth.
        """
        for neighbor in self.graph.successors(current_node):
            if neighbor not in visited:
                self.subgraph.add_node(neighbor)
                self.subgraph.add_edge(
                    current_node, 
                    neighbor, 
                    **self.graph.get_edge_attributes(current_node, neighbor)
                )
                visited.add(neighbor)
                queue.append((neighbor, current_distance + 1))
        for predecessor in self.graph.predecessors(current_node):
            if predecessor not in visited:
                self.subgraph.add_node(predecessor)
                self.subgraph.add_edge(
                    predecessor,
                    current_node, 
                    **self.graph.get_edge_attributes(predecessor, current_node)
                )
                visited.add(predecessor)
                queue.append((predecessor, current_distance + 1))

    def _build_neighbor_subgraph(self, nodes: List[str], n: int = 2) -> None:
        """
        Builds a subgraph containing up to n-order neighbors for the given nodes.

        Args:
            nodes: List of nodes to start from.
            n: Depth of neighbor traversal.
        """
        self.subgraph = self.graph.subgraph(nodes)
        for node in nodes:
            queue = [(node, 0)]
            visited = {node}
            while queue:
                current_node, current_distance = queue.pop(0)
                if current_distance >= n:
                    continue
                self._add_neighbors_to_subgraph(current_node, visited, queue, current_distance)

    def _extract_edges_with_attributes(self) -> List[Tuple[Any, Any, Any]]:
        """
        Efficiently extracts (source, relation, target) triples from the current subgraph.
        """
        return [
            (u, data.get('relation'), v)
            for u, v, data in self.subgraph.get_edges()
        ]

    def _extract_entities_from_question(self, question: str) -> List[str]:
        """
        Extracts entities from a question using the LLM.

        Args:
            question: The input question string.

        Returns:
            List of extracted entities.
        """
        prompt = (
            "Extract all named entities from the following question. "
            "Return a comma-separated list of entities only, no explanations or extra text.\n"
            f"Question: {question}"
        )
        message = [{"role": "system", "content": "You are a helpful AI assistant."}]
        entity_response = self.llm.chat(prompt, message, llm_config=self.llm_config)
        entities = [e.strip() for e in entity_response.split(',') if len(e.strip()) > 0]
        return entities
