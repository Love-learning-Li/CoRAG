import re
import threading
import time
import os

from copy import deepcopy
from typing import Any, Optional, List, Dict, Tuple
from datasets import Dataset

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from logger_config import logger
from vllm_client import VllmClient
from search.search_utils import extract_retrieved_documents, search_with_variants
from data_utils import parse_answer_logprobs
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
    subquery = re.sub(r'^(SubQuery|Query)\s*:\s*', '', subquery, flags=re.IGNORECASE).strip()

    return subquery, thought


def _normalize_subanswer(subanswer: str) -> str:
    if not subanswer:
        return ''

    cleaned = re.sub(r'<think>.*?</think>', '', subanswer, flags=re.DOTALL).strip()
    for marker in ['Final Answer:', 'SubAnswer:', 'Answer:']:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()

    cleaned = re.sub(r'^(SubAnswer|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE).strip()
    if cleaned.lower() in {'yes', 'no', 'insufficient information'}:
        return cleaned.lower()
    if not cleaned or cleaned.lower() in {'no relevant information found', 'unable to determine.', 'unable to determine', 'cannot determine', 'unknown'}:
        return 'No relevant information found'
    return cleaned


def _extract_final_answer(raw_answer: str) -> str:
    if not raw_answer:
        return ''

    text = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
    if 'Final Answer:' in text:
        text = text.split('Final Answer:')[-1].strip()

    text = re.sub(
        r'\b(SubQuery|SubAnswer|Intermediate query|Intermediate answer)\s*:\s*',
        '',
        text,
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]

    text = re.sub(r'^(Final Answer|Answer)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    lowered = text.lower().strip()
    if lowered in {'unable to determine.', 'unable to determine', 'cannot determine', 'unknown'}:
        return 'No relevant information found'
    if 'no relevant information found' in lowered:
        return 'No relevant information found'
    if lowered in {'yes', 'no', 'insufficient information'}:
        return lowered
    return text


def _is_no_info_answer(text: str) -> bool:
    return _normalize_subanswer(text) == 'No relevant information found'


def _sanitize_followup_subquery(subquery: str) -> str:
    normalized = subquery.strip()
    normalized = re.sub(r'^Who is the nationality of (?P<entity>.+?)\?$', r'What nationality is \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who is the country of origin for (?P<entity>.+?)\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who is the country of (?P<entity>.+?)\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^What country is the origin of (?:(?:the|a) )?(?:movie|film|song|album|show|series)\s+(?P<entity>.+?)\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^What country did (?:(?:the|a) )?(?:movie|film|song|album|show|series)\s+(?P<entity>.+?) originate from\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who was the father of (?P<entity>.+?)\?$', r'Who is the father of \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who was the mother of (?P<entity>.+?)\?$', r'Who is the mother of \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who was (?P<entity>.+?)\'s (?P<relation>husband|wife|spouse|father|mother)\?$', r'Who is \g<entity>\'s \g<relation>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who is the child-in-law of (?P<entity>.+?)\?$', r'Who is \g<entity>\'s child?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _clean_answer_entity(answer: str) -> str:
    cleaned = _normalize_subanswer(answer)
    if not cleaned or cleaned == 'No relevant information found':
        return ''
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip('.').strip()
    for separator in [';', ' or ', ' and ', ', and ', ',']:
        if separator in cleaned:
            cleaned = cleaned.split(separator)[0].strip()
    cleaned = re.sub(
        r'^(?:Queen|King|Emperor|Empress|Prince|Princess|Grand\s+Duke|Grand\s+Duchess|'
        r'Duke|Duchess|Count|Countess|Earl|Baron|Baroness|Lord|Lady|Sir|Dame|Saint|St\.)\s+',
        '',
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _extract_comparison_entities(query: str) -> Optional[Tuple[str, str, str]]:
    attr = _comparison_attribute_from_query(query)
    if not attr:
        return None
    cleaned_query = query.strip()

    role_match = re.match(
        r'^(?:Are|Do|Did|Were)\s+'
        r'(?P<left_prefix>director|composer|performer)\s+of\s+(?:the\s+)?(?P<left_kind>film|movie|song|track|single)\s+(?P<left>.+?)\s+and\s+'
        r'(?P<right_prefix>director|composer|performer)\s+of\s+(?:the\s+)?(?P<right_kind>film|movie|song|track|single)\s+(?P<right>.+?)\s+'
        r'(?:from|of|born in|share|sharing|have|has|originat(?:e|ed))\b.*\?$',
        cleaned_query,
        flags=re.IGNORECASE,
    )
    if role_match and role_match.group('left_prefix').lower() == role_match.group('right_prefix').lower():
        left = role_match.group('left').strip().strip(' ,?')
        right = role_match.group('right').strip().strip(' ,?')
        return left, right, attr

    role_tail_match = re.match(
        r'^(?:Do|Are|Did|Were)\s+both\s+'
        r'(?P<left>.+?)\s+and\s+(?P<right>.+?)\s+'
        r'(?:films|movies|songs|tracks|singles)\s+have\s+the\s+'
        r'(?P<role>directors?|composers?|performers?)\b.*\?$',
        cleaned_query,
        flags=re.IGNORECASE,
    )
    if role_tail_match:
        left = role_tail_match.group('left').strip().strip(' ,')
        right = role_tail_match.group('right').strip().strip(' ,')
        return left, right, attr

    plain_patterns = [
        r'^(?:Are both|Were both)\s+(?:(?:the|both)\s+)?(?:movies|films|songs|tracks|singles)?\,?\s*(?P<body>.+?)\s+(?:from|of|born in)\s+the same .*\?$',
        r'^(?:Did|Do)\s+(?:the\s+)?(?:movies|films|songs|tracks|singles)?\s*(?P<body>.+?)\s+(?:originate|originated|come)\s+from\s+the same .*\?$',
        r'^(?:Did|Do)\s+both\s+(?P<body>.+?)\s+(?:originate|originated|come)\s+from\s+the same .*\?$',
        r'^(?:Are|Do|Did|Were)\s+(?P<body>.+?)\s+of the same .*\?$',
        r'^(?:Do|Are|Did|Were)\s+both\s+(?P<body>.+?)\s+.*same .*\?$',
    ]
    for pattern in plain_patterns:
        match = re.match(pattern, cleaned_query, flags=re.IGNORECASE)
        if not match:
            continue
        body = match.group('body').strip().strip(' ,')
        split = re.split(r'\s+and\s+', body, maxsplit=1, flags=re.IGNORECASE)
        if len(split) != 2:
            continue
        left = re.sub(r'^(?:movies|films|songs|tracks|singles)\s*,?\s*', '', split[0], flags=re.IGNORECASE).strip(' ,')
        right = re.sub(r'^(?:movies|films|songs|tracks|singles)\s*,?\s*', '', split[1], flags=re.IGNORECASE).strip(' ,')
        right = re.sub(r'\s+(?:movies|films|songs|tracks|singles)\b$', '', right, flags=re.IGNORECASE).strip(' ,')
        right = re.sub(r'\s+(?:have|that|who)\b.*$', '', right, flags=re.IGNORECASE).strip(' ,')
        if left and right:
            return left, right, attr

    return None


def _attribute_query_for_entity(entity: str, attribute: str) -> Optional[str]:
    entity = entity.strip()
    if not entity:
        return None
    if attribute == 'nationality':
        return f"What nationality is {entity}?"
    if attribute == 'country':
        return f"Which country is {entity} from?"
    if attribute == 'place':
        return f"Where was {entity} born?"
    return None


def _comparison_attribute_from_query(query: str) -> Optional[str]:
    lowered = query.lower()
    if 'same nationality' in lowered:
        return 'nationality'
    if 'same country' in lowered:
        return 'country'
    if 'same place' in lowered or 'born in the same' in lowered:
        return 'place'
    return None


def _comparison_role_media_type(query: str, role: str) -> Optional[str]:
    lowered = query.lower()
    role_token = role.lower()
    role_context_pattern = re.compile(
        rf'{role_token}\s+of\s+(?:the\s+)?(?P<media_type>film|movie|song|track|single|album|show|play|series)\b',
        flags=re.IGNORECASE,
    )
    match = role_context_pattern.search(query)
    if match:
        return _normalize_media_type(match.group('media_type'), role_token)
    if role_token == 'performer':
        return 'song'
    if role_token == 'director':
        return 'film'
    if role_token == 'composer':
        if any(token in lowered for token in [' song ', ' songs', ' track ', ' single ']):
            return 'song'
        return 'film'
    return None


def _extract_media_type(query: str) -> Optional[str]:
    lowered = query.lower()
    for candidate in ('film', 'movie', 'song', 'track', 'single', 'album', 'show', 'play', 'series'):
        if re.search(rf'\b{candidate}\b', lowered):
            return candidate
    return None


def _normalize_media_type(media_type: Optional[str], role: Optional[str] = None) -> str:
    normalized = (media_type or '').lower()
    if normalized in {'movie', 'film'}:
        return 'film'
    if normalized in {'song', 'track', 'single'}:
        return 'song'
    if normalized in {'album', 'show', 'play', 'series'}:
        return normalized
    if role == 'performer':
        return 'song'
    return 'film'


def _extract_role_target(query: str) -> Optional[Tuple[str, str, str]]:
    patterns = [
        r'^(?:Who directed|Who is the director of) (?:(?:the|a) )?(?P<media_type>film|movie|show|play|series) (?P<title>.+?)\?$',
        r'^(?:Who directed) the (?P<year>(?:18|19|20)\d{2}) (?P<media_type>film|movie|show|play|series) (?P<title>.+?)\?$',
        r'^Who composed (?:the music for )?(?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?)\?$',
        r'^Who performed (?:(?:the|a) )?(?P<media_type>song|track|single) (?P<title>.+?)\?$',
        r'^Who is the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?)\?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        role = match.groupdict().get('role')
        title = match.groupdict().get('title')
        media_type = _normalize_media_type(match.groupdict().get('media_type'))
        lowered = query.lower()
        if role:
            return role.lower(), title.strip(), media_type
        if 'directed' in lowered:
            return 'director', title.strip(), media_type
        if 'composed' in lowered:
            return 'composer', title.strip(), media_type
        if 'performed' in lowered:
            return 'performer', title.strip(), media_type
    return None


def _role_query(role: str, title: str, media_type: Optional[str] = None) -> str:
    role = role.lower()
    media_type = _normalize_media_type(media_type, role)
    if role == 'director':
        target_type = 'film' if media_type in {'film', 'show', 'play', 'series'} else media_type
        return f'Who directed the {target_type} {title}?'
    if role == 'composer':
        if media_type == 'song':
            return f'Who composed the song {title}?'
        target_type = 'film' if media_type in {'film', 'movie'} else media_type
        return f'Who composed the music for the {target_type} {title}?'
    if role == 'performer':
        return f'Who performed the song {title}?'
    return f'Who is the {role} of the {media_type} {title}?'


def _extract_role_attribute_question(query: str) -> Optional[Tuple[str, str, str, str]]:
    patterns = [
        r'^What nationality is the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?)\?$',
        r'^Which country is the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?) from\?$',
        r'^Where was the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?) born\?$',
        r'^When was the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?) born\?$',
        r'^What is the date of birth of the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?)\?$',
        r'^Where did the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?) die\?$',
        r'^When did the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?) die\?$',
        r'^What is the date of death of the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series) (?P<title>.+?)\?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        lowered = query.lower()
        if 'nationality' in lowered:
            attr = 'nationality'
        elif 'from' in lowered:
            attr = 'country'
        elif 'date of birth' in lowered or 'when was' in lowered:
            attr = 'date_of_birth'
        elif 'born' in lowered:
            attr = 'place_of_birth'
        elif 'date of death' in lowered or 'when did' in lowered:
            attr = 'date_of_death'
        else:
            attr = 'place_of_death'
        return (
            match.group('role').lower(),
            match.group('title').strip(),
            attr,
            _normalize_media_type(match.group('media_type'), match.group('role').lower()),
        )
    return None


def _attribute_query(entity: str, attr: str) -> Optional[str]:
    entity = entity.strip()
    if not entity:
        return None
    mapping = {
        'nationality': f'What nationality is {entity}?',
        'country': f'Which country is {entity} from?',
        'place': f'Where was {entity} born?',
        'place_of_birth': f'Where was {entity} born?',
        'date_of_birth': f'What is the date of birth of {entity}?',
        'place_of_death': f'Where did {entity} die?',
        'date_of_death': f'What is the date of death of {entity}?',
    }
    return mapping.get(attr)


def _extract_possessive_relation_attribute(query: str) -> Optional[Tuple[str, str, str]]:
    patterns = [
        r'^What is the date of birth of (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter)\?$',
        r'^What is the date of death of (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter)\?$',
        r'^Where was (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter) born\?$',
        r'^Where did (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter) die\?$',
        r'^What nationality is (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter)\?$',
        r'^Which country is (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter) from\?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        lowered = query.lower()
        if 'date of birth' in lowered:
            attr = 'date_of_birth'
        elif 'date of death' in lowered:
            attr = 'date_of_death'
        elif 'born' in lowered:
            attr = 'place_of_birth'
        elif 'die' in lowered:
            attr = 'place_of_death'
        elif 'nationality' in lowered:
            attr = 'nationality'
        else:
            attr = 'country'
        return match.group('entity').strip(), match.group('relation').strip().lower(), attr
    return None


def _extract_direct_relation_query(query: str) -> Optional[Tuple[str, str]]:
    patterns = [
        r'^Who is (?P<entity>.+?)\'s (?P<relation>father|mother|husband|wife|spouse|child|son|daughter)\?$',
        r'^Who is the (?P<relation>father|mother|husband|wife|spouse|child|son|daughter) of (?P<entity>.+?)\?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if match:
            entity = match.group('entity').strip()
            lowered_entity = entity.lower()
            if any(token in lowered_entity for token in ['director of', 'composer of', 'performer of', 'author of', 'writer of']):
                return None
            return entity, match.group('relation').strip().lower()
    return None


def _extract_role_relation_question(query: str) -> Optional[Tuple[str, str, str, str]]:
    patterns = [
        r'^Who is the (?P<relation>father|mother|child|son|daughter|spouse|husband|wife) of the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series)\s+(?P<title>.+?)\?$',
        r'^Who is the (?P<relation>child|son|daughter) of the (?P<role>director|composer|performer) of (?:(?:the|a) )?(?P<media_type>film|movie|song|album|show|play|series)\s+(?P<title>.+?)\?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if match:
            return (
                match.group('relation').lower(),
                match.group('role').lower(),
                match.group('title').strip(),
                _normalize_media_type(match.group('media_type'), match.group('role').lower()),
            )
    return None


class CoRagAgent:

    def __init__(
            self, vllm_client: VllmClient, corpus: Dataset,
            graph_api_url: Optional[str] = None,
            tokenizer: Optional[PreTrainedTokenizerFast] = None,
            final_vllm_client: Optional[VllmClient] = None,
            sub_answer_vllm_client: Optional[VllmClient] = None,
            retrieval_max_variants: int = 4,
            retrieval_per_query_limit: int = 5,
            retrieval_service_top_k: int = 10,
            enable_deterministic_planner: bool = True,
            enable_generic_deterministic_rules: bool = True,
            enable_dataset_specific_rules: bool = False,
            dataset_rule_profile: str = "none",
    ):
        self.vllm_client = vllm_client
        self.final_vllm_client = final_vllm_client
        self.sub_answer_vllm_client = sub_answer_vllm_client
        self.corpus = corpus
        self.graph_api_url = graph_api_url
        self.retrieval_max_variants = max(1, retrieval_max_variants)
        self.retrieval_per_query_limit = max(1, retrieval_per_query_limit)
        self.retrieval_service_top_k = max(self.retrieval_per_query_limit, retrieval_service_top_k)
        self.enable_deterministic_planner = enable_deterministic_planner
        self.enable_generic_deterministic_rules = enable_generic_deterministic_rules
        self.enable_dataset_specific_rules = enable_dataset_specific_rules
        self.dataset_rule_profile = (dataset_rule_profile or "none").lower()

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

    def _minimum_required_steps(self, query: str, max_path_length: int) -> int:
        required = max_path_length
        if self.enable_generic_deterministic_rules:
            required = max(required, self._minimum_generic_steps(query))
        if self._dataset_specific_rules_enabled():
            required = max(required, self._minimum_dataset_specific_steps(query))
        return required

    def _dataset_specific_rules_enabled(self) -> bool:
        return self.enable_dataset_specific_rules and self.dataset_rule_profile == "2wiki"

    def _minimum_generic_steps(self, query: str) -> int:
        required = 1
        if _extract_role_relation_question(query):
            required = max(required, 2)
        if _extract_role_attribute_question(query):
            required = max(required, 2)
        if _extract_possessive_relation_attribute(query):
            required = max(required, 2)
        comparison = _extract_comparison_entities(query)
        if comparison:
            left, right, _ = comparison
            role_based = any(token in query.lower() for token in ['director', 'composer', 'performer'])
            required = max(required, 4 if role_based else 2)
            if not left or not right:
                required = max(required, 1)
        return required

    def _minimum_dataset_specific_steps(self, query: str) -> int:
        if re.search(r'father-in-law|mother-in-law|maternal grandmother|maternal grandfather|paternal grandmother|paternal grandfather|stepmother|stepfather', query, flags=re.IGNORECASE):
            return 2
        return 1

    def _plan_generic_subquery(self, query: str, past_subqueries: List[str], past_subanswers: List[str]) -> Optional[str]:
        step = len(past_subqueries)

        role_relation = _extract_role_relation_question(query)
        if role_relation:
            relation, role, title, media_type = role_relation
            if step == 0:
                return _role_query(role, title, media_type)
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s {relation}?"

        role_attribute = _extract_role_attribute_question(query)
        if role_attribute:
            role, title, attr, media_type = role_attribute
            if step == 0:
                return _role_query(role, title, media_type)
            if step == 1:
                entity = _clean_answer_entity(past_subanswers[0]) if past_subanswers else ''
                return _attribute_query(entity, attr)

        comparison = _extract_comparison_entities(query)
        if comparison:
            left, right, attr = comparison
            lowered = query.lower()
            role = None
            if 'director' in lowered:
                role = 'director'
            elif 'composer' in lowered:
                role = 'composer'
            elif 'performer' in lowered:
                role = 'performer'
            shared_media_type = _comparison_role_media_type(query, role) if role else None
            left_media_type = _extract_media_type(left) or shared_media_type
            right_media_type = _extract_media_type(right) or shared_media_type
            if role:
                if step == 0:
                    return _role_query(role, left, left_media_type)
                if step == 1:
                    return _role_query(role, right, right_media_type)
                if step == 2 and len(past_subanswers) >= 1:
                    return _attribute_query(_clean_answer_entity(past_subanswers[0]), attr)
                if step == 3 and len(past_subanswers) >= 2:
                    return _attribute_query(_clean_answer_entity(past_subanswers[1]), attr)
            else:
                left_query = _attribute_query(left, attr)
                right_query = _attribute_query(right, attr)
                if step == 0:
                    return left_query
                if step == 1:
                    return right_query

        relation_attr = _extract_possessive_relation_attribute(query)
        if relation_attr:
            entity, relation, attr = relation_attr
            if step == 0:
                return f"Who is {entity}'s {relation}?"
            if step == 1 and past_subanswers:
                return _attribute_query(_clean_answer_entity(past_subanswers[0]), attr)

        direct_relation = _extract_direct_relation_query(query)
        if direct_relation:
            entity, relation = direct_relation
            if step == 0:
                return f"Who is {entity}'s {relation}?"

        return None

    def _plan_dataset_specific_subquery(self, query: str, past_subqueries: List[str], past_subanswers: List[str]) -> Optional[str]:
        step = len(past_subqueries)

        if re.search(r'father-in-law', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the father-in-law of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s spouse?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the spouse of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s father?"
            if step == 2 and len(past_subanswers) >= 2 and _is_no_info_answer(past_subanswers[1]):
                return f"Who was {subject}'s spouse?"

        if re.search(r'mother-in-law', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the mother-in-law of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s spouse?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the spouse of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s mother?"
            if step == 2 and len(past_subanswers) >= 2 and _is_no_info_answer(past_subanswers[1]):
                return f"Who was {subject}'s spouse?"

        if re.search(r'maternal grandmother', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the maternal grandmother of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s mother?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the mother of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s mother?"

        if re.search(r'maternal grandfather', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the maternal grandfather of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s mother?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the mother of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s father?"

        if re.search(r'paternal grandmother', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the paternal grandmother of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s father?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the father of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s mother?"

        if re.search(r'paternal grandfather', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is the paternal grandfather of (.+?)\?$', r'\1', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s father?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the father of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s father?"

        if re.search(r'stepmother', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is (?P<entity>.+?)\'s stepmother\?$', r'\g<entity>', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s father?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the father of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s spouse?"

        if re.search(r'stepfather', query, flags=re.IGNORECASE):
            subject = re.sub(r'^Who is (?P<entity>.+?)\'s stepfather\?$', r'\g<entity>', query, flags=re.IGNORECASE).strip()
            if step == 0:
                return f"Who is {subject}'s mother?"
            if step == 1 and past_subanswers and _is_no_info_answer(past_subanswers[0]):
                return f"Who is the mother of {subject}?"
            if step == 1 and past_subanswers:
                return f"Who is {_clean_answer_entity(past_subanswers[0])}'s spouse?"

        return None

    def _plan_next_subquery(self, query: str, past_subqueries: List[str], past_subanswers: List[str]) -> Optional[Tuple[str, str]]:
        if not self.enable_deterministic_planner:
            return None

        if self.enable_generic_deterministic_rules:
            generic = self._plan_generic_subquery(query, past_subqueries, past_subanswers)
            if generic:
                return generic, 'generic deterministic planner'

        if self._dataset_specific_rules_enabled():
            dataset_specific = self._plan_dataset_specific_subquery(query, past_subqueries, past_subanswers)
            if dataset_specific:
                return dataset_specific, f'{self.dataset_rule_profile} deterministic planner'

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
        past_retrieval_stats: List[Dict[str, Any]] = kwargs.pop('past_retrieval_stats', [])
        assert len(past_subqueries) == len(past_subanswers) == len(past_doc_ids)
        if past_documents:
            assert len(past_documents) == len(past_doc_ids)
        if past_thoughts:
            assert len(past_thoughts) == len(past_subqueries)
        if past_retrieval_stats:
            assert len(past_retrieval_stats) == len(past_subqueries)

        effective_max_path_length = self._minimum_required_steps(query, max_path_length)
        subquery_temp: float = temperature
        num_llm_calls: int = 0
        max_num_llm_calls: int = 4 * max(1, effective_max_path_length - len(past_subqueries))
        while len(past_subqueries) < effective_max_path_length:
            planned = self._plan_next_subquery(query, past_subqueries, past_subanswers)
            planned_subquery, planner_label = planned if planned else (None, None)
            thought = planner_label
            if planned_subquery:
                subquery = _sanitize_followup_subquery(planned_subquery)
            else:
                if num_llm_calls >= max_num_llm_calls:
                    break
                num_llm_calls += 1
                messages: List[Dict] = get_generate_subquery_prompt(
                    query=query,
                    past_subqueries=past_subqueries,
                    past_subanswers=past_subanswers,
                    task_desc=task_desc,
                )
                self._truncate_long_messages(messages, max_length=max_message_length)

                subquery_resp = self.vllm_client.call_chat(messages=messages, temperature=subquery_temp, **kwargs)
                if isinstance(subquery_resp, dict):
                    subquery_raw = str(subquery_resp.get('choices', [{}])[0].get('message', {}).get('content', ''))
                else:
                    subquery_raw = str(subquery_resp)
                subquery, thought = _normalize_subquery(subquery_raw)
                subquery = _sanitize_followup_subquery(subquery)

            # Only honour [STOP] after at least 1 sub-query has been completed.
            # At step-0 the LLM has seen "Nothing yet" as context and may incorrectly
            # output [STOP], which would skip all retrieval.
            if len(past_subqueries) >= 1 and subquery.strip().upper() == '[STOP]':
                logger.debug("LLM issued [STOP] signal — ending subquery loop early.")
                break

            if not subquery:
                subquery_temp = max(subquery_temp, 0.7)
                continue

            if subquery in past_subqueries:
                if planned_subquery:
                    break
                subquery_temp = max(subquery_temp, 0.7)
                continue

            subquery_temp = temperature
            subanswer, doc_ids, documents, retrieval_stats = self._get_subanswer_and_doc_ids(
                subquery=subquery, max_message_length=max_message_length
            )

            past_subqueries.append(subquery)
            past_subanswers.append(subanswer)
            past_doc_ids.append(doc_ids)
            past_documents.append(documents)
            past_thoughts.append(thought or '')
            past_retrieval_stats.append(retrieval_stats)

        return RagPath(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            past_doc_ids=past_doc_ids,
            past_documents=past_documents,
            past_thoughts=past_thoughts,
            past_retrieval_stats=past_retrieval_stats,
        )

    def generate_final_answer(
            self, corag_sample: RagPath, task_desc: str,
            max_message_length: int = 4096,
            documents: Optional[List[str]] = None, **kwargs
    ) -> str:
        normalized_subanswers = [_normalize_subanswer(sa or '') for sa in (corag_sample.past_subanswers or [])]
        # If every intermediate step failed retrieval, avoid parametric hallucination.
        if normalized_subanswers and all(answer == 'No relevant information found' for answer in normalized_subanswers):
            return 'No relevant information found'
        comparison = _extract_comparison_entities(corag_sample.query)
        if comparison:
            non_empty = [sa for sa in normalized_subanswers if sa and sa != 'No relevant information found']
            if len(non_empty) < 2:
                return 'insufficient information'
        else:
            required_fact_steps = 1
            if _extract_role_relation_question(corag_sample.query) or _extract_role_attribute_question(corag_sample.query):
                required_fact_steps = 2
            elif _extract_possessive_relation_attribute(corag_sample.query):
                required_fact_steps = 2
            elif self._dataset_specific_rules_enabled() and self._minimum_dataset_specific_steps(corag_sample.query) > 1:
                required_fact_steps = 2

            informative_answers = [sa for sa in normalized_subanswers if sa and sa != 'No relevant information found']
            if normalized_subanswers and len(informative_answers) < required_fact_steps:
                return 'No relevant information found'

        messages: List[Dict] = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        client = self.final_vllm_client if self.final_vllm_client else self.vllm_client
        raw_resp = client.call_chat(messages=messages, **kwargs)
        if isinstance(raw_resp, dict):
            raw_answer = str(raw_resp.get('choices', [{}])[0].get('message', {}).get('content', ''))
        else:
            raw_answer = str(raw_resp)
        return _extract_final_answer(raw_answer)

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

        completion_resp = self.vllm_client.call_chat(messages=messages, return_str=False, n=int(1.5 * n), **kwargs)
        completion: Dict = completion_resp if isinstance(completion_resp, dict) else {'choices': []}
        # We process the subqueries to extract thoughts but sample_subqueries only returns the query strings
        subqueries_and_thoughts: List[Tuple[str, Optional[str]]] = [_normalize_subquery(c['message']['content']) for c in completion['choices']]
        subqueries: List[str] = [_sanitize_followup_subquery(s[0]) for s in subqueries_and_thoughts if s[0]]
        subqueries = list(set(subqueries))[:n]

        return subqueries

    def _get_subanswer_and_doc_ids(
            self, subquery: str, max_message_length: int = 4096
    ) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        _t_retr = time.time()
        retriever_results = search_with_variants(
            query=subquery,
            graph_api_url=self.graph_api_url,
            max_variants=self.retrieval_max_variants,
            per_query_limit=self.retrieval_per_query_limit,
            service_top_k=self.retrieval_service_top_k,
        )
        _elapsed_retr = time.time() - _t_retr
        logger.info(
            f"[RETRIEVAL] subquery={repr(subquery)[:80]} "
            f"→ {len(retriever_results)} result(s) in {_elapsed_retr:.1f}s"
        )

        doc_ids, documents, format_issue_count = extract_retrieved_documents(
            retriever_results,
            reverse_order=True,
            corpus=self.corpus,
        )
        retrieval_stats = {
            "subquery": subquery,
            "raw_result_count": len(retriever_results),
            "usable_result_count": len(documents),
            "format_issue_count": format_issue_count,
            "elapsed_seconds": _elapsed_retr,
        }

        # If retrieval returned nothing, skip the LLM call entirely.
        # Calling the LLM with an empty context causes it to answer from parametric
        # memory (hallucination), producing plausible-sounding but wrong entities that
        # pollute subsequent subquery steps and the final answer.  A consistent
        # "No relevant information found" signal is far less harmful than a confident
        # hallucinated entity like "Joan Baez" or "Ferdinand I of Austria".
        if not documents:
            logger.debug(f"No documents retrieved for subquery: {subquery!r}. Skipping sub-answer LLM call.")
            return "No relevant information found", [], [], retrieval_stats

        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=subquery,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        client = self.sub_answer_vllm_client if self.sub_answer_vllm_client else self.vllm_client
        subanswer_resp = client.call_chat(messages=messages, temperature=0., max_tokens=128)
        if isinstance(subanswer_resp, dict):
            subanswer_raw = str(subanswer_resp.get('choices', [{}])[0].get('message', {}).get('content', ''))
        else:
            subanswer_raw = str(subanswer_resp)
        subanswer = _normalize_subanswer(subanswer_raw)
        return subanswer, doc_ids, documents, retrieval_stats

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
                    new_candidate.past_subqueries = new_candidate.past_subqueries or []
                    new_candidate.past_subanswers = new_candidate.past_subanswers or []
                    new_candidate.past_doc_ids = new_candidate.past_doc_ids or []
                    new_candidate.past_documents = new_candidate.past_documents or []
                    new_candidate.past_thoughts = new_candidate.past_thoughts or []
                    new_candidate.past_subqueries.append(subquery)
                    new_candidate.past_thoughts.append('')  # Thoughts are not currently captured in tree search expansion
                    subanswer, doc_ids, documents, retrieval_stats = self._get_subanswer_and_doc_ids(
                        subquery=subquery, max_message_length=max_message_length
                    )
                    new_candidate.past_subanswers.append(subanswer)
                    new_candidate.past_doc_ids.append(doc_ids)
                    new_candidate.past_documents.append(documents)
                    new_candidate.past_retrieval_stats = new_candidate.past_retrieval_stats or []
                    new_candidate.past_retrieval_stats.append(retrieval_stats)
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
        past_subqueries = current_path.past_subqueries or []
        past_subanswers = current_path.past_subanswers or []
        messages: List[Dict] = get_generate_intermediate_answer_prompt(
            subquery=current_path.query,
            documents=[f'Q: {q}\nA: {a}' for q, a in zip(past_subqueries, past_subanswers)],
        )
        messages.append({'role': 'assistant', 'content': 'No relevant information found'})
        self._truncate_long_messages(messages, max_length=max_message_length)

        response_raw = self.vllm_client.call_chat(
            messages=messages,
            return_str=False,
            max_tokens=1,
            extra_body={
                "prompt_logprobs": 1
            }
        )
        response: Dict = response_raw if isinstance(response_raw, dict) else {'choices': []}
        answer_logprobs: List[float] = parse_answer_logprobs(response)

        return sum(answer_logprobs) / len(answer_logprobs)

    def _eval_state_without_answer(
            self, path: RagPath, num_rollouts: int, task_desc: str,
            max_path_length: int = 3,
            temperature: float = 0.7,
            max_message_length: int = 4096
    ) -> float:
        assert len(path.past_subqueries or []) > 0
        if num_rollouts <= 0:
            return self._eval_single_path(path)

        rollout_paths: List[RagPath] = []
        for _ in range(num_rollouts):
            rollout_path: RagPath = self.sample_path(
                query=path.query, task_desc=task_desc,
                max_path_length=min(max_path_length, len(path.past_subqueries or []) + 2), # rollout at most 2 steps
                temperature=temperature, max_message_length=max_message_length,
                past_subqueries=deepcopy(path.past_subqueries),
                past_subanswers=deepcopy(path.past_subanswers),
                past_doc_ids=deepcopy(path.past_doc_ids),
                past_documents=deepcopy(path.past_documents),
                past_thoughts=deepcopy(path.past_thoughts),
                past_retrieval_stats=deepcopy(path.past_retrieval_stats),
            )
            rollout_paths.append(rollout_path)

        scores: List[float] = [self._eval_single_path(p) for p in rollout_paths]
        # TODO: should we use the min score instead?
        return sum(scores) / len(scores)
