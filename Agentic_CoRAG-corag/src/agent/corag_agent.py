import re
import threading
import os

from copy import deepcopy
from typing import Optional, List, Dict, Tuple
from datasets import Dataset

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from search.search_utils import search_with_variants
from data_utils import format_input_context, parse_answer_logprobs
from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt, get_generate_final_answer_prompt
from agent.agent_utils import RagPath
from utils import batch_truncate


def _normalize_subquery(subquery: str) -> Tuple[str, Optional[str]]:
    # Extract think blocks
    thought_match = re.search(r'<think>(.*?)</think>', subquery, flags=re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else None
    
    # Remove think blocks
    subquery = re.sub(r'<think>.*?</think>', '', subquery, flags=re.DOTALL)

    subquery = subquery.strip()
    if subquery.startswith('"') and subquery.endswith('"'):
        subquery = subquery[1:-1]
    if subquery.startswith('Intermediate query'):
        subquery = re.sub(r'^Intermediate query \d+: ', '', subquery)

    return subquery, thought


def _clean_answer_entity(answer: str) -> str:
    cleaned = re.sub(r'\s+', ' ', answer).strip().strip('.').strip()
    if not cleaned:
        return ''
    if cleaned.lower().startswith('no relevant information found'):
        return ''
    separators = [';', ' or ', ' and ', ', and ', ',']
    for separator in separators:
        if separator in cleaned:
            cleaned = cleaned.split(separator)[0].strip()
    # Strip leading honorific titles so subqueries use bare entity names
    honorific_prefix = (
        r'^(?:Queen|King|Emperor|Empress|Prince|Princess|Grand\s+Duke|Grand\s+Duchess|'
        r'Duke|Duchess|Count|Earl|Baron|Baroness|Lord|Lady|Dame|Sir|'
        r'Tsar|Tsarina|Sultan|Pope|Saint|St\.)\s+'
    )
    cleaned = re.sub(honorific_prefix, '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


class CoRagAgent:

    def __init__(
            self, vllm_client: VllmClient, corpus: Dataset,
            graph_api_url: Optional[str] = None,
            tokenizer: Optional[PreTrainedTokenizerFast] = None,
            final_vllm_client: Optional[VllmClient] = None,
            sub_answer_vllm_client: Optional[VllmClient] = None
    ):
        self.vllm_client = vllm_client
        self.final_vllm_client = final_vllm_client
        self.sub_answer_vllm_client = sub_answer_vllm_client
        self.corpus = corpus
        self.graph_api_url = graph_api_url

        # Tokenizer may be local even if vLLM model points to a remote machine path.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None
            model_ref = getattr(vllm_client, "model", "")
            if isinstance(model_ref, str):
                model_ref = model_ref.strip()

            try:
                # 1) If model_ref is a local dir on THIS machine, load from it.
                if isinstance(model_ref, str) and os.path.isdir(model_ref):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_ref, use_fast=True, local_files_only=True
                    )
                # 2) If model_ref looks like an absolute path but doesn't exist locally,
                #    it's likely a remote path (e.g., vLLM server path). Do NOT treat as HF repo id.
                elif isinstance(model_ref, str) and model_ref.startswith("/"):
                    logger.warning(
                        f"vllm_client.model={model_ref!r} looks like an absolute path but is not a local directory. "
                        f"Likely a remote vLLM server path. Please pass a local tokenizer to CoRagAgent "
                        f"(e.g., /home/models/Qwen3-8B) via the `tokenizer` parameter."
                    )
                    self.tokenizer = None
                # 3) Otherwise assume it's a HF repo id.
                elif isinstance(model_ref, str) and model_ref:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True)
            except Exception as e:
                logger.warning(f"Failed to init tokenizer from model_ref={model_ref!r}: {e}. Proceeding without tokenizer.")
                self.tokenizer = None
                
        self.lock = threading.Lock()

    def _plan_next_subquery(self, query: str, past_subanswers: List[str]) -> Optional[str]:
        cleaned_query = query.strip()

        # All rule-based patterns below are strictly 2-hop decompositions.
        # If we have already completed 2 or more hops, do NOT fire the rule planner
        # again — doing so would incorrectly treat the step-2 answer as a new pivot
        # entity and generate a spurious step-3 subquery.
        if len(past_subanswers) >= 2:
            return None

        two_hop_patterns = [
            (r"^Who is the father-in-law of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s spouse?" if not resolved else f"Who is {resolved}'s father?"),
            (r"^Who is (?P<entity>.+?)'s father-in-law\?$", lambda entity, resolved: f"Who is {entity}'s spouse?" if not resolved else f"Who is {resolved}'s father?"),
            (r"^Who is the maternal grandmother of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s mother?" if not resolved else f"Who is {resolved}'s mother?"),
            (r"^Who is the paternal grandmother of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s father?" if not resolved else f"Who is {resolved}'s mother?"),
            (r"^Who is the maternal grandfather of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s mother?" if not resolved else f"Who is {resolved}'s father?"),
            (r"^Who is the paternal grandfather of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s father?" if not resolved else f"Who is {resolved}'s father?"),
            (r"^Who is (?P<entity>.+?)'s stepmother\?$", lambda entity, resolved: f"Who is {entity}'s father?" if not resolved else f"Who is {resolved}'s spouse?"),
            (r"^Who is the stepmother of (?P<entity>.+?)\?$", lambda entity, resolved: f"Who is {entity}'s father?" if not resolved else f"Who is {resolved}'s spouse?"),
            (r"^Who is the child of the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Who is {resolved}'s child?"),
            (r"^What is the date of birth of (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife)\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"What is the date of birth of {resolved}?"),
            (r"^What is the date of death of (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife)\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"What is the date of death of {resolved}?"),
            (r"^Where did (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife) die\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"Where did {resolved} die?"),
            (r"^Where was the place of death of (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife)\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"Where did {resolved} die?"),
            (r"^Where was the place of birth of (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife)\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"Where was {resolved} born?"),
            (r"^Where does (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife) work at\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"Where does {resolved} work at?"),
            (r"^What nationality is (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife)\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"What nationality is {resolved}?"),
            (r"^Which country (?P<entity>.+?)'s (?P<relation>father|mother|husband|wife) is from\?$", lambda entity, resolved, relation=None: f"Who is {entity}'s {relation}?" if not resolved else f"Which country {resolved} is from?"),
            (r"^Which award the (?P<role>.+?) of (?P<entity>.+?) received\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Which award {resolved} received?"),
            (r"^What is the award that the (?P<role>.+?) of (?P<entity>.+?) (?:received|earned|won|got)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Which award {resolved} received?"),
            # --- Missing patterns for "[attribute] of [role] of [entity]" ---
            (r"^What nationality is the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"What nationality is {resolved}?"),
            (r"^Which country (?:is )?the (?P<role>.+?) of (?P<entity>.+?) (?:is )?from\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Which country is {resolved} from?"),
            (r"^What is the date of birth of the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"What is the date of birth of {resolved}?"),
            (r"^What is the date of death of the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"What is the date of death of {resolved}?"),
            (r"^Where did the (?P<role>.+?) of (?P<entity>.+?) die\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Where did {resolved} die?"),
            (r"^Where was the (?P<role>.+?) of (?P<entity>.+?) born\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Where was {resolved} born?"),
            (r"^Where does the (?P<role>.+?) of (?P<entity>.+?) work(?: at)?\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Where does {resolved} work at?"),
            (r"^Where was the place of birth of the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Where was {resolved} born?"),
            (r"^Where was the place of death of the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda entity, resolved, role=None: f"Who is the {role} of {entity}?" if not resolved else f"Where did {resolved} die?"),
        ]

        resolved_entity = _clean_answer_entity(past_subanswers[-1]) if past_subanswers else ''

        for pattern, builder in two_hop_patterns:
            match = re.match(pattern, cleaned_query, flags=re.IGNORECASE)
            if not match:
                continue
            groups = match.groupdict()
            entity = groups.get('entity', '').strip()
            if 'role' in groups:
                return builder(entity, resolved_entity, groups['role'].strip())
            if 'relation' in groups:
                return builder(entity, resolved_entity, groups['relation'].strip())
            return builder(entity, resolved_entity)

        return None

    def sample_path(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            **kwargs
    ) -> RagPath:
        past_subqueries: List[str] = kwargs.pop('past_subqueries', [])
        past_subanswers: List[str] = kwargs.pop('past_subanswers', [])
        past_doc_ids: List[List[str]] = kwargs.pop('past_doc_ids', [])
        past_documents: List[List[str]] = kwargs.pop('past_documents', [])
        past_thoughts: List[str] = kwargs.pop('past_thoughts', [])
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)
        if past_documents:
            assert len(past_documents) == len(past_doc_ids)
        if past_thoughts:
            assert len(past_thoughts) == len(past_subqueries)

        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 4 * (max_path_length - len(past_subqueries))
        while len(past_subqueries) < max_path_length and num_llm_calls < max_num_llm_calls:
            planned_subquery = self._plan_next_subquery(query=query, past_subanswers=past_subanswers)
            if planned_subquery and planned_subquery not in past_subqueries:
                subquery = planned_subquery
                thought = None
            else:
                num_llm_calls += 1
                messages: List[Dict] = get_generate_subquery_prompt(
                    query=query,
                    past_subqueries=past_subqueries,
                    past_subanswers=past_subanswers,
                    task_desc=task_desc,
                )
                self._truncate_long_messages(messages, max_length=max_message_length)

                subquery_raw: str = self.vllm_client.call_chat(messages=messages, temperature=subquery_temp, **kwargs)
                subquery, thought = _normalize_subquery(subquery_raw)

                # Early-stop: LLM signals that enough information has been gathered
                if subquery.strip().upper() == '[STOP]':
                    logger.debug("LLM issued [STOP] signal — ending subquery loop early.")
                    break

            if subquery in past_subqueries:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids, documents = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)
            past_documents.append(documents)
            past_thoughts.append(thought)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
            past_documents=past_documents,
            past_thoughts=past_thoughts,
        )

    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        client = self.final_vllm_client if self.final_vllm_client else self.vllm_client
        return client.call_chat(messages=messages, **kwargs)

    def _truncate_long_messages(self, messages: List[Dict], max_length: int):
        if self.tokenizer is None:
            return

        for msg in messages:
            if len(msg['content']) < 2 * max_length:
                continue

            with self.lock:
                msg['content'] = batch_truncate(
                    [msg['content']], tokenizer=self.tokenizer, max_length=max_length, truncate_from_middle=True
                )[0]

    def sample_subqueries(self, query: str, task_desc: str, n: int = 10, max_message_length: int = 4096, **kwargs) -> List[str]:
        messages: List[Dict] = get_generate_subquery_prompt(
            query=query,
            past_subqueries=kwargs.pop('past_subqueries', []),
            past_subanswers=kwargs.pop('past_subanswers', []),
            task_desc=task_desc,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        completion: Dict = self.vllm_client.call_chat(messages=messages, return_str=False, n=int(1.5 * n), **kwargs)
        # We process the subqueries to extract thoughts but sample_subqueries only returns the query strings
        subqueries_and_thoughts: List[Tuple[str, str]] = [_normalize_subquery(c['message']['content']) for c in completion['choices']]
        subqueries: List[str] = [s[0] for s in subqueries_and_thoughts]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List, List[str]]:
        retriever_results = search_with_variants(
            query=subquery,
            graph_api_url=self.graph_api_url,
            max_variants=4,
            per_query_limit=5,
        )

        if self.graph_api_url:
            documents = []
            doc_ids = []
            for res in retriever_results:
                if isinstance(res, str):
                    documents.append(res)
                    doc_ids.append('graph_chunk')
                elif isinstance(res, dict):
                    content = res.get('contents') or res.get('content') or res.get('text') or str(res)
                    documents.append(content)
                    doc_ids.append(str(res.get('id') or res.get('doc_id') or 'graph_chunk'))
            documents = documents[::-1]
        else:
            doc_ids: List[str] = [res['doc_id'] for res in retriever_results]
            documents: List[str] = [format_input_context(self.corpus[int(doc_id)]) for doc_id in doc_ids][::-1]
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        client = self.sub_answer_vllm_client if self.sub_answer_vllm_client else self.vllm_client
        subanswer: str = client.call_chat(messages=messages, temperature=0., max_tokens=128)
        return subanswer, doc_ids, documents

    def tree_search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        return self._search(
            query=query, task_desc=task_desc,
            max_path_length=max_path_length,
            max_message_length=max_message_length,
            temperature=temperature,
            expand_size=expand_size, num_rollouts=num_rollouts, beam_size=beam_size,
            **kwargs
        )

    def best_of_n(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            n: int = 4,
            **kwargs
    ) -> RagPath:
        sampled_paths: List[RagPath] = []
        for idx in range(n):
            path: RagPath = self.sample_path(
                query=query, task_desc=task_desc,
                max_path_length=max_path_length,
                max_message_length=max_message_length,
                temperature=0. if idx == 0 else temperature,
                **kwargs
            )
            sampled_paths.append(path)

        scores: List[float] = [self._eval_single_path(p) for p in sampled_paths]
        return sampled_paths[scores.index(min(scores))]

    def _search(
            self, query: str, task_desc: str,
            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            expand_size: int = 4, num_rollouts: int = 2, beam_size: int = 1,
            **kwargs
    ) -> RagPath:
        candidates: List[RagPath] = [RagPath(query=query, past_subqueries=[], past_subanswers=[], past_doc_ids=[], past_documents=[], past_thoughts=[])]
        for step in range(max_path_length):
            new_candidates: List[RagPath] = []
            for candidate in candidates:
                new_subqueries: List[str] = self.sample_subqueries(
                    query=query, task_desc=task_desc,
                    past_subqueries=deepcopy(candidate.past_subqueries),
                    past_subanswers=deepcopy(candidate.past_subanswers),
                    n=expand_size, temperature=temperature, max_message_length=max_message_length
                )
                for subquery in new_subqueries:
                    new_candidate: RagPath = deepcopy(candidate)
                    new_candidate.past_subqueries.append(subquery)
                    new_candidate.past_thoughts.append(None) # Thoughts are not currently captured in tree search expansion
                    subanswer, doc_ids, documents = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidate.past_documents.append(documents)
                    new_candidates.append(new_candidate)

            if len(new_candidates) > beam_size:
                scores: List[float] = []
                for path in new_candidates:
                    score = self._eval_state_without_answer(
                        path, num_rollouts=num_rollouts,
                        task_desc=task_desc,
                        max_path_length=max_path_length,
                        temperature=temperature,
                        max_message_length=max_message_length
                    )
                    scores.append(score)

                # lower scores are better
                new_candidates = [c for c, s in sorted(zip(new_candidates, scores), key=lambda x: x[1])][:beam_size]

            candidates = new_candidates

        return candidates[0]

    def _eval_single_path(self, current_path: RagPath, max_message_length: int = 4096) -> float:
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(current_path.past_subqueries, current_path.past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response: Dict = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=1,
            extra_body={
                "prompt_logprobs": 1
            }
        )
        answer_logprobs: List[float] = parse_answer_logprobs(response)

        return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        assert len(path.past_subqueries) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries) + 2), # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
                past_documents=deepcopy(path.past_documents),
                past_thoughts=deepcopy(path.past_thoughts),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        # TODO: should we use the min score instead?
        return sum(scores) / len(scores)
