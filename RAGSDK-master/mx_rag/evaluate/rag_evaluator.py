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

from typing import Dict, Any, Optional, List
import os
import asyncio
import logging
import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM
from loguru import logger
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    AnswerSimilarity,
    AnswerRelevancy,
    AnswerAccuracy,
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall,
    ContextUtilization,
    ContextRelevance,
    Faithfulness,
    NoiseSensitivity,
    ResponseGroundedness
)

from mx_rag.utils.common import validate_params, validate_list_str, validate_list_list_str, TEXT_MAX_LEN, MB
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils.file_check import SecDirCheck

# Disable logs from ragas
if os.environ.get("DISABLE_RAGAS_LOGGING", "1") == "1":
    logging.getLogger("ragas").setLevel(logging.CRITICAL)


class RAGEvaluator:
    RAG_METRICS = {
        "answer_correctness": AnswerCorrectness(),
        "answer_similarity": AnswerSimilarity(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "context_entity_recall": ContextEntityRecall(),
        "context_utilization": ContextUtilization(),
        "faithfulness": Faithfulness(),
        "noise_sensitivity": NoiseSensitivity(),
        "nv_response_groundedness": ResponseGroundedness(),
        "nv_accuracy": AnswerAccuracy(),
        "nv_context_relevance": ContextRelevance()
    }

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, LLM), message="param must be instance of LLM"),
        embeddings=dict(validator=lambda x: isinstance(x, (TextEmbedding, TEIEmbedding, Embeddings)),
                        message="param must be instance of TextEmbedding, TEIEmbedding or Embeddings")
    )
    def __init__(self, llm: LLM, embeddings: Embeddings):
        self.embeddings = embeddings
        self.evaluator_llm = LangchainLLMWrapper(llm)

    @staticmethod
    def _check_metrics(metrics: List[str], metrics_dict: dict) -> List:
        # Remove unsupported metrics and warn
        supported = []
        unsupported = []
        seen = set()
        for m in metrics:
            if m not in metrics_dict:
                unsupported.append(m)
            elif m not in seen:
                supported.append(metrics_dict[m])
                seen.add(m)
        if unsupported:
            logger.warning(f"Unsupported metrics removed: {unsupported}. Supported: {list(metrics_dict.keys())}")
        if len(supported) < len(metrics):
            logger.warning("Duplicate metric names detected and removed.")
        if not supported:
            raise ValueError("No valid metrics provided after filtering unsupported and duplicates.")
        return supported

    @staticmethod
    def _check_dataset(dataset: Dict[str, Any]) -> bool:
        check_attribute = {
            "user_input": lambda x: validate_list_str(x, [0, 128], [1, TEXT_MAX_LEN]),
            "response": lambda x: validate_list_str(x, [0, 128], [1, TEXT_MAX_LEN]),
            "retrieved_contexts": lambda x: validate_list_list_str(x, [0, 128], [1, 128], [1, TEXT_MAX_LEN]),
            "reference": lambda x: validate_list_str(x, [0, 128], [1, TEXT_MAX_LEN])
        }
        if not (isinstance(dataset, dict) and all(isinstance(key, str) for key in dataset)):
            logger.error(
                f"Dataset validation failed: dataset should be a dict with all string keys. "
                f"Received type: {type(dataset)}, keys: {list(dataset.keys()) if isinstance(dataset, dict) else 'N/A'}"
            )
            return False
        if not 1 <= len(dataset) <= len(check_attribute):
            logger.error(f"Dataset validation failed: number of keys {len(dataset)} is out of allowed range [1, 4]. ")
            return False
        return all(key in check_attribute and check_attribute.get(key)(value) for key, value in dataset.items())

    @staticmethod
    def _load_metric_prompts(metric, language, prompts_path):
        if isinstance(metric, AnswerSimilarity):
            # AnswerSimilarity object has no attribute 'load_prompts'
            return
        SecDirCheck(prompts_path, 4 * MB).check()
        try:
            new_prompts = metric.load_prompts(prompts_path, language)
            if new_prompts:
                metric.set_prompts(**new_prompts)
        except FileNotFoundError:
            logger.warning(f"Prompt file not found for metric '{metric.name}'. Using default.")
        except Exception as e:
            logger.error(f"Failed to load prompts for metric '{metric.name}': {e}")

    @validate_params(
        metrics=dict(validator=lambda x: validate_list_str(x, [1, len(RAGEvaluator.RAG_METRICS)], [1, 50]),
                     message="param must be List[str], list length range [1, 14], str length range [1, 50]"),
        dataset=dict(
            validator=lambda x: RAGEvaluator._check_dataset(x), message="param check error detail see log"),
        language=dict(
            validator=lambda x: x is None or (isinstance(x, str) and x in ("chinese", "english")),
            message="param must be None or 'chinese' or 'english'"),
        prompts_path=dict(
            validator=lambda x: x is None or (isinstance(x, str) and 0 < len(x) < 256),
            message="param must be None or str, and str length range [1, 255]"),
        show_progress=dict(validator=lambda x: isinstance(x, bool), message="param must be a boolean")
    )
    def evaluate(
            self,
            metrics: List[str],
            dataset: Dict[str, Any],
            language: Optional[str] = None,
            prompts_path: Optional[str] = None,
            show_progress: bool = False
    ) -> Optional[Dict[str, List[float]]]:
        """
        Evaluate the given dataset using specified metrics.
        Args:
            metrics: List of metric names to use for evaluation.
            dataset: Dataset in dict format (see ragas docs for structure).
            language: Target language ('chinese' or 'english').
            prompts_path: Path to directory containing custom prompts.
            show_progress: Whether to show the progress bar during evaluation. Default is False.
        Returns:
            Dict[str, List[float]]: Scores for each metric, or None if evaluation fails.
        """
        metrics = self._check_metrics(metrics, self.RAG_METRICS)
        if prompts_path is None:
            prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

        if language and language != "english":
            logger.info(f"Adapting prompts in metrics for language '{language}'")
            self._adapt_metrics(metrics, language, prompts_path)

        evaluation_dataset = Dataset.from_dict(dataset)
        try:
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                embeddings=self.embeddings,
                show_progress=show_progress
            )
        except KeyError as e:
            logger.error(f"ragas evaluate run failed: {e}")
            return None
        except ValueError as e:
            logger.error(f"ragas evaluate run failed: {e}")
            return None
        except Exception as e:
            logger.error(f"ragas evaluate run failed: {e}")
            return None
        # ragas 0.2.x returns a EvaluationResult object with .scores as a list
        df = pd.DataFrame(result.scores)
        return df.to_dict(orient='list')

    def _adapt_metrics(self, metrics: List, language: str, prompts_path: Optional[str]) -> None:
        """
        Adapts the prompts in the metrics to the given language.
        Runs LLM-based adaptations concurrently for performance.
        """
        async_tasks = []
        async_metrics = []

        for metric in metrics:
            if isinstance(metric, (AnswerAccuracy, ContextRelevance, ResponseGroundedness)):
                continue
            if prompts_path:
                # Handle file-based loading synchronously
                self._load_metric_prompts(metric, language, prompts_path)
            else:
                # Collect async tasks for LLM-based adaptation
                async_tasks.append(metric.adapt_prompts(language, self.evaluator_llm))
                async_metrics.append(metric)

        if not async_tasks:
            return

        # Run all LLM-based adaptations concurrently
        async def run_adaptations():
            return await asyncio.gather(*async_tasks, return_exceptions=True)

        try:
            results = asyncio.run(run_adaptations())
            for metric, result in zip(async_metrics, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to adapt metric '{metric.name}' via LLM: {result}")
                elif result:
                    metric.set_prompts(**result)
        except asyncio.TimeoutError as e:
            logger.error(f"Operation timed out: {e}")
        except MemoryError as e:
            logger.error(f"Memory error occurred: {e}")
        except Exception as e:
            # This catches errors in asyncio.run() itself, e.g., if no event loop can be started.
            logger.error(f"Unexpected error during concurrent metric adaptation: {e}")
