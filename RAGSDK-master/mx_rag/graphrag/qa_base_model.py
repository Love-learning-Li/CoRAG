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


from typing import List, Optional
from tqdm import tqdm

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.graphrag.prompts.evaluate_qa import (
    EVAL_GENERATION_TEMPLATE_CN, EVAL_GENERATION_TEMPLATE_EN, LLM_PLAIN_TEMPLATE
)
from mx_rag.utils.common import Lang


class EvaluationStrategy:
    """
    Abstract base class for evaluation strategies.

    Subclasses should implement the evaluate method to provide specific evaluation logic.
    """

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        responses: List[str],
        language: Lang = Lang.CH
    ) -> List[str]:
        """
        Evaluate model responses.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of reference answers.
            responses (List[str]): List of model responses.
            language (Lang, optional): Language for evaluation. Defaults to Lang.CH.

        Returns:
            List[str]: Evaluation results.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError(
            "Subclasses of EvaluationStrategy must implement the evaluate method."
        )


class GenerationEvaluationStrategy(EvaluationStrategy):
    """
    Evaluation strategy for generation metrics using a language model.

    Attributes:
        llm (Text2TextLLM): The language model instance.
        llm_config (LLMParameterConfig): Configuration for the language model.
    """

    def __init__(self, llm: Text2TextLLM, llm_config: LLMParameterConfig) -> None:
        """
        Initialize the GenerationEvaluationStrategy.

        Args:
            llm (Text2TextLLM): The language model instance.
            llm_config (LLMParameterConfig): Configuration for the language model.
        """
        self.llm = llm
        self.llm_config = llm_config

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        responses: List[str],
        language: Lang = Lang.CH
    ) -> List[str]:
        """
        Evaluate responses using the generation evaluation template.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of reference answers.
            responses (List[str]): List of model responses.
            language (Lang, optional): Language for evaluation. Defaults to Lang.CH.

        Returns:
            List[str]: Evaluation results.
        """
        eval_results = []
        for question, answer, response in tqdm(
            zip(questions, answers, responses), total=len(questions), desc="Evaluting Answers"
        ):
            prompt = self._build_prompt(question, answer, response, language)
            sys_messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            eval_result = self.llm.chat(prompt, sys_messages, llm_config=self.llm_config)
            eval_results.append(eval_result)
        return eval_results

    def _build_prompt(
        self,
        question: str,
        answer: str,
        response: str,
        language: Lang
    ) -> str:
        """
        Build the evaluation prompt based on the language.

        Args:
            question (str): The question.
            answer (str): The reference answer.
            response (str): The model response.
            language (Lang): The language for the prompt.

        Returns:
            str: The formatted evaluation prompt.
        """
        template = EVAL_GENERATION_TEMPLATE_CN if language == Lang.CH else EVAL_GENERATION_TEMPLATE_EN
        return template.format(question=question, answer=answer, response=response)


class QABaseModel:
    """
    Base class for Question Answering models.

    Attributes:
        llm (Text2TextLLM): The language model instance.
        llm_config (LLMParameterConfig): Configuration for the language model.
        metric (str): The evaluation metric.
        evaluation_strategy (Optional[EvaluationStrategy]): The evaluation strategy instance.
    """

    def __init__(
        self,
        llm: Text2TextLLM,
        llm_config: LLMParameterConfig,
        metric: str = "generation"
    ) -> None:
        """
        Initialize the QABaseModel.

        Args:
            llm (Text2TextLLM): The language model instance.
            llm_config (LLMParameterConfig): Configuration for the language model.
            metric (str, optional): The evaluation metric. Defaults to "generation".
        """
        self.llm = llm
        self.llm_config = llm_config
        self.metric = metric
        self.evaluation_strategy = self._select_evaluation_strategy()

    def generate(self, questions: List[str]) -> List[str]:
        """
        Generate responses for a list of questions using the language model.

        Args:
            questions (List[str]): List of questions.

        Returns:
            List[str]: Generated responses.
        """
        return self._plain_generate(questions)

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        responses: List[str],
        language: Lang = Lang.CH
    ) -> List[str]:
        """
        Evaluate model responses using the selected evaluation strategy.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of reference answers.
            responses (List[str]): List of model responses.
            language (Lang, optional): Language for evaluation. Defaults to Lang.CH.

        Returns:
            List[str]: Evaluation results.

        Raises:
            ValueError: If the metric is not supported.
        """
        if not self.evaluation_strategy:
            raise ValueError(f"Metric '{self.metric}' is not supported.")
        return self.evaluation_strategy.evaluate(questions, answers, responses, language)
    
    def _select_evaluation_strategy(self) -> Optional[EvaluationStrategy]:
        """
        Select the appropriate evaluation strategy based on the metric.

        Returns:
            Optional[EvaluationStrategy]: The evaluation strategy instance, or None if not supported.
        """
        if self.metric == "generation":
            return GenerationEvaluationStrategy(self.llm, self.llm_config)
        # Extend here for additional strategies.
        return None

    def _plain_generate(self, questions: List[str]) -> List[str]:
        """
        Generate plain responses for a list of questions.

        Args:
            questions (List[str]): List of questions.

        Returns:
            List[str]: Generated responses.
        """
        responses = []
        for question in questions:
            sys_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that answers questions as simply as possible."
                    ),
                }
            ]
            prompt = LLM_PLAIN_TEMPLATE.format(question=question)
            response = self.llm.chat(prompt, sys_messages, llm_config=self.llm_config)
            responses.append(response)
        return responses
