import re

import requests

from typing import Any, Callable, Dict, List, Optional, Tuple

from logger_config import logger


CONTAINER_KEYS = (
    'chunks', 'chunk_list', 'data', 'results', 'docs', 'documents', 'passages',
    'items', 'records', 'hits',
)
CONTENT_KEYS = (
    'contents', 'content', 'text', 'snippet', 'passage', 'chunk_text', 'body',
    'document', 'document_text',
)
TITLE_KEYS = ('title', 'name', 'doc_title', 'article_title')
DOC_ID_KEYS = ('doc_id', 'id', 'chunk_id', 'document_id', 'passage_id')


def _flatten_result_nodes(node: Any) -> List[Any]:
    if isinstance(node, list):
        flattened: List[Any] = []
        for item in node:
            flattened.extend(_flatten_result_nodes(item))
        return flattened

    if isinstance(node, dict):
        for key in CONTAINER_KEYS:
            value = node.get(key)
            if isinstance(value, (list, dict)):
                return _flatten_result_nodes(value)
        return [node] if node else []

    return [node] if node else []


def _normalize_graph_api_results(results: Any) -> List[Any]:
    """Normalize Graph API responses into a flat list for downstream consumers."""
    return _flatten_result_nodes(results)


def _extract_first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_extract_first_text(item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in CONTENT_KEYS:
            candidate = _extract_first_text(value.get(key))
            if candidate:
                return candidate
        for key in TITLE_KEYS:
            candidate = _extract_first_text(value.get(key))
            if candidate:
                return candidate
    return ""


def _extract_doc_id(record: Dict[str, Any]) -> str:
    for key in DOC_ID_KEYS:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "graph_chunk"


def normalize_retrieval_record(result: Any) -> Tuple[Optional[Dict[str, Any]], bool]:
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return None, True
        return {"doc_id": "graph_chunk", "title": "", "contents": text}, False

    if not isinstance(result, dict):
        return None, True

    title = ""
    for key in TITLE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            title = value.strip()
            break

    contents = ""
    for key in CONTENT_KEYS:
        contents = _extract_first_text(result.get(key))
        if contents:
            break

    if not contents and title:
        contents = title

    doc_id = _extract_doc_id(result)
    if not contents and doc_id == "graph_chunk":
        return None, True

    return {
        "doc_id": doc_id,
        "title": title,
        "contents": contents,
    }, False


def normalize_retrieval_results(results: List[Any]) -> Tuple[List[Dict[str, Any]], int]:
    normalized: List[Dict[str, Any]] = []
    format_issue_count = 0
    for result in results:
        record, had_issue = normalize_retrieval_record(result)
        if had_issue:
            format_issue_count += 1
        if record:
            normalized.append(record)
    return normalized, format_issue_count


def extract_retrieved_documents(
        results: List[Any],
        reverse_order: bool = False,
        limit: int = 0,
        corpus: Any = None,
) -> Tuple[List[str], List[str], int]:
    normalized_results, format_issue_count = normalize_retrieval_results(results)
    doc_ids: List[str] = []
    documents: List[str] = []

    for result in normalized_results:
        contents = result["contents"]
        if not contents and corpus is not None:
            doc_id = result["doc_id"]
            if str(doc_id).isdigit():
                from data_utils import format_input_context
                contents = format_input_context(corpus[int(doc_id)])
        if not contents:
            format_issue_count += 1
            continue
        doc_ids.append(result["doc_id"])
        documents.append(contents)

    if reverse_order:
        doc_ids = list(reversed(doc_ids))
        documents = list(reversed(documents))

    if limit and limit > 0:
        doc_ids = doc_ids[:limit]
        documents = documents[:limit]

    return doc_ids, documents, format_issue_count


def _canonicalize_query_text(query: str) -> str:
    normalized = re.sub(r'<think>.*?</think>', ' ', query, flags=re.DOTALL)
    prefix_pattern = re.compile(
        r'^\s*(?:SubQuery|SubAnswer|Final Answer|Intermediate query \d+)\s*:\s*',
        flags=re.IGNORECASE,
    )
    while True:
        updated = prefix_pattern.sub('', normalized)
        if updated == normalized:
            break
        normalized = updated

    normalized = normalized.strip()
    normalized = re.sub(r'^Who is the nationality of (?P<entity>.+?)\?$', r'What nationality is \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who is the country of origin for (?P<entity>.+?)\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who is the country of (?P<entity>.+?)\?$', r'Which country is \g<entity> from?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who was the father of (?P<entity>.+?)\?$', r'Who is the father of \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^Who was the mother of (?P<entity>.+?)\?$', r'Who is the mother of \g<entity>?', normalized, flags=re.IGNORECASE)
    normalized = normalized.replace('“', '"').replace('”', '"').replace('’', "'")
    normalized = re.sub(r'\s+', ' ', normalized.replace('"', '')).strip()
    return normalized


def _append_variant(variants: List[str], candidate: Optional[str]) -> None:
    if not candidate:
        return
    normalized = _canonicalize_query_text(candidate)
    if normalized and normalized not in variants:
        variants.append(normalized)


def _append_title_variants(variants: List[str], text: str) -> None:
    paren_stripped = re.sub(r'\s*\([^)]*\)', '', text).strip()
    year_stripped = re.sub(r'\b(?:18|19|20)\d{2}\b', '', paren_stripped).strip()
    compact = re.sub(r'\s{2,}', ' ', year_stripped)
    if compact and compact != text:
        _append_variant(variants, compact)


def build_search_queries(query: str, max_variants: int = 4) -> List[str]:
    normalized_query = _canonicalize_query_text(query)
    variants: List[str] = []
    rewrite_variants: List[str] = []

    patterns = [
        (r"^Who was (?P<entity>.+?) born in\?$", lambda m: [f"{m.group('entity')} place of birth", f"Where was {m.group('entity')} born?"]),
        (r"^Who did (?P<entity>.+?) die in\?$", lambda m: [f"{m.group('entity')} place of death", f"Where did {m.group('entity')} die?"]),
        (r"^Where was (?P<entity>.+?) born\?$", lambda m: [f"{m.group('entity')} place of birth", f"birth place {m.group('entity')}"]),
        (r"^Where did (?P<entity>.+?) die\?$", lambda m: [f"{m.group('entity')} place of death", f"death place {m.group('entity')}"]),
        (r"^When was (?P<entity>.+?) born\?$", lambda m: [f"{m.group('entity')} date of birth", f"birth date {m.group('entity')}"]),
        (r"^When did (?P<entity>.+?) die\?$", lambda m: [f"{m.group('entity')} date of death", f"death date {m.group('entity')}"]),
        (r"^What is the date of birth of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} date of birth", f"birth date {m.group('entity')}"]),
        (r"^What is the date of death of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} date of death", f"death date {m.group('entity')}"]),
        (r"^What nationality is (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} nationality", f"nationality {m.group('entity')}"]),
        (r"^Which country is (?P<entity>.+?) from\?$", lambda m: [f"{m.group('entity')} country", f"country {m.group('entity')}"]),
        (r"^What country is (?:(?:the|a) )?(?:(?:film|song|movie|album|show|play|series)\s+)?(?P<entity>.+?) from\?$", lambda m: [f"{m.group('entity')} country", f"country {m.group('entity')}"]),
        (r"^Which country (?:(?:the|a) )?(?:(?:film|song|movie|album|show|play|series)\s+)?(?P<entity>.+?) is from\?$", lambda m: [f"{m.group('entity')} country", f"country {m.group('entity')}"]),
        (r"^Where does (?P<entity>.+?) work at\?$", lambda m: [f"{m.group('entity')} employer", f"{m.group('entity')} workplace"]),
        (r"^Which award (?P<entity>.+?) (?:received|earned|won|got)\?$", lambda m: [f"{m.group('entity')} award", f"awards of {m.group('entity')}"]),
        (r"^Who is (?P<entity>.+?)'s (?P<relation>father|mother|spouse|husband|wife|child|son|daughter|father-in-law|mother-in-law|uncle|aunt|stepmother|stepfather)\?$", lambda m: [f"{m.group('entity')} {m.group('relation')}", f"{m.group('relation')} of {m.group('entity')}"]),
        (r"^Who is the (?P<relation>father|mother|spouse|husband|wife|child|son|daughter|father-in-law|mother-in-law|uncle|aunt|stepmother|stepfather|maternal grandmother|maternal grandfather|paternal grandmother|paternal grandfather) of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} {m.group('relation')}", f"{m.group('relation')} of {m.group('entity')}"]),
        (r"^Who directed the (?P<year>(?:18|19|20)\d{2}) (?P<kind>film|movie|show|play|series) (?P<title>.+?)\?$", lambda m: [f"{m.group('title')} {m.group('kind')} director", f"{m.group('title')} director", f"director {m.group('title')}"]),
        (r"^Who performed (?:(?:the|a) )?(?:song|track|single) (?P<title>.+?)\?$", lambda m: [f"{m.group('title')} performer", f"performer {m.group('title')}", f"{m.group('title')} originally performed by"]),
        (r"^Who composed (?:the music for )?(?:(?:the|a) )?(?:song|track|single) (?P<title>.+?)\?$", lambda m: [f"{m.group('title')} composer", f"composer {m.group('title')}", f"songwriter {m.group('title')}", f"written by {m.group('title')}"]),
        (r"^Who composed (?:the music for )?(?:(?:the|a) )?(?:film|movie|album|show|play|series) (?P<title>.+?)\?$", lambda m: [f"{m.group('title')} composer", f"composer {m.group('title')}", f"music by {m.group('title')}"]),
        (r"^Who directed (?:(?:the|a) )?(?:film|movie|show|play|series) (?P<title>.+?)\?$", lambda m: [f"{m.group('title')} director", f"director {m.group('title')}", f"directed by {m.group('title')}"]),
        (r"^Who is the (?P<role>.+?) of (?:(?:the|a) )?(?:film|song|movie|album|show|play|series)\s+(?P<title>.+?)\?$", lambda m: [f"{m.group('title')} {m.group('role')}", f"{m.group('role')} {m.group('title')}"]),
        (r"^Who is the (?P<role>.+?) of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} {m.group('role')}", f"{m.group('role')} {m.group('entity')}"]),
        (r"^What year was (?:(?:the|a) )?(?:film|movie|show|play|series) (?P<title>.+?) released\?$", lambda m: [f"{m.group('title')} release year", f"{m.group('title')} released", f"release date {m.group('title')}"]),
    ]

    for pattern, builder in patterns:
        match = re.match(pattern, normalized_query, flags=re.IGNORECASE)
        if not match:
            continue
        for variant in builder(match):
            _append_variant(rewrite_variants, variant)
            _append_title_variants(rewrite_variants, variant)
        break

    possessive_match = re.search(r"(?P<entity>.+?)'s (?P<relation>.+)", normalized_query)
    if possessive_match:
        _append_variant(rewrite_variants, f"{possessive_match.group('entity')} {possessive_match.group('relation')}")
        _append_variant(rewrite_variants, f"{possessive_match.group('relation')} of {possessive_match.group('entity')}")

    _append_title_variants(rewrite_variants, normalized_query)

    for variant in rewrite_variants:
        _append_variant(variants, variant)
    _append_variant(variants, normalized_query)

    return variants[:max_variants]


def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for result in results:
        key = result.get('doc_id') or result.get('contents') or result.get('title')
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def search_by_http(query: str, host: str = 'localhost', port: int = 8090, top_k: int = 10) -> List[Any]:
    url = f"http://{host}:{port}"
    try:
        response = requests.post(url, json={'query': query, 'top_k': top_k}, timeout=(3, 10))
        if response.status_code == 200:
            return _normalize_graph_api_results(response.json())
        logger.error(f"Failed to get a response. Status code: {response.status_code}")
        return []
    except requests.RequestException as e:
        logger.error(f"HTTP search request failed: {e}")
        return []


def search_by_graph_api(query: str, url: str, top_k: int = 10) -> List[Any]:
    _read_timeouts = [240, 480]
    last_exc = None
    for attempt, read_timeout in enumerate(_read_timeouts):
        try:
            response = requests.post(
                url,
                json={'query': query, 'top_k': top_k},
                headers={"Content-Type": "application/json"},
                timeout=(5, read_timeout),
            )
            if response.status_code == 200:
                return _normalize_graph_api_results(response.json())
            logger.error(f"Failed to get a response from graph API. Status code: {response.status_code}")
            return []
        except requests.Timeout as e:
            last_exc = e
            logger.warning(
                f"Graph API read timeout (attempt {attempt + 1}/{len(_read_timeouts)}, "
                f"read_timeout={read_timeout}s) for query={repr(query)[:80]}: {e}"
            )
            if attempt + 1 < len(_read_timeouts):
                logger.info("Retrying graph API call with doubled timeout...")
        except requests.RequestException as e:
            logger.error(f"Error calling graph API: {e}")
            return []
    logger.error(
        f"Graph API timed out after {len(_read_timeouts)} attempts "
        f"for query={repr(query)[:80]}: {last_exc}"
    )
    return []


def search_with_variants(
        query: str,
        graph_api_url: Optional[str] = None,
        host: str = 'localhost',
        port: int = 8090,
        max_variants: int = 4,
        per_query_limit: int = 5,
        service_top_k: int = 10,
) -> List[Dict[str, Any]]:
    queries = build_search_queries(query, max_variants=max_variants)
    merged_results: List[Dict[str, Any]] = []

    search_fn: Callable[..., List[Any]]
    search_kwargs: Dict[str, object]
    if graph_api_url:
        search_fn = search_by_graph_api
        search_kwargs = {'url': graph_api_url, 'top_k': service_top_k}
    else:
        search_fn = search_by_http
        search_kwargs = {'host': host, 'port': port, 'top_k': service_top_k}

    failed_variants = 0
    for candidate_query in queries:
        try:
            raw_results = search_fn(query=candidate_query, **search_kwargs)
        except Exception as e:
            failed_variants += 1
            logger.warning(f"Search failed for variant '{candidate_query}': {e}")
            continue

        normalized_results, format_issue_count = normalize_retrieval_results(raw_results)
        if format_issue_count:
            logger.warning(
                f"Retriever returned {format_issue_count} malformed result(s) for variant "
                f"{candidate_query!r}; ignored during normalization."
            )
        if per_query_limit > 0:
            normalized_results = normalized_results[:per_query_limit]
        merged_results.extend(normalized_results)

    if failed_variants == len(queries):
        logger.warning(
            f"All {failed_variants} search variant(s) failed for query: {query!r}. "
            "Check that the retrieval service is reachable."
        )
    return _dedupe_results(merged_results)
