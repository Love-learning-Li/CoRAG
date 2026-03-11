import re

import requests

from typing import List, Dict, Callable, Optional

from logger_config import logger


def _normalize_graph_api_results(results):
    """Normalize Graph API responses into a list for downstream consumers."""
    if isinstance(results, dict):
        for key in ['chunks', 'data', 'results', 'docs', 'passages']:
            value = results.get(key)
            if isinstance(value, list):
                return value
        return [results] if results else []
    if isinstance(results, list):
        return results
    return []


def _canonicalize_query_text(query: str) -> str:
    return re.sub(r'\s+', ' ', query.replace('"', '').replace("'", '')).strip()


def _append_variant(variants: List[str], candidate: Optional[str]) -> None:
    if not candidate:
        return
    normalized = _canonicalize_query_text(candidate)
    if normalized and normalized not in variants:
        variants.append(normalized)


def build_search_queries(query: str, max_variants: int = 4) -> List[str]:
    variants: List[str] = []
    normalized_query = _canonicalize_query_text(query)
    _append_variant(variants, normalized_query)

    patterns = [
        (r"^Where was (?P<entity>.+?) born\?$", lambda m: [f"{m.group('entity')} place of birth", f"birth place {m.group('entity')}"]),
        (r"^Where did (?P<entity>.+?) die\?$", lambda m: [f"{m.group('entity')} place of death", f"death place {m.group('entity')}"]),
        (r"^When was (?P<entity>.+?) born\?$", lambda m: [f"{m.group('entity')} date of birth", f"birth date {m.group('entity')}"]),
        (r"^When did (?P<entity>.+?) die\?$", lambda m: [f"{m.group('entity')} date of death", f"death date {m.group('entity')}"]),
        (r"^What is the date of birth of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} date of birth", f"birth date {m.group('entity')}"]),
        (r"^What is the date of death of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} date of death", f"death date {m.group('entity')}"]),
        (r"^What nationality is (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} nationality", f"{m.group('entity')} country"]),
        (r"^Which country (?P<entity>.+?) is from\?$", lambda m: [f"{m.group('entity')} country", f"{m.group('entity')} nationality"]),
        (r"^Where does (?P<entity>.+?) work at\?$", lambda m: [f"{m.group('entity')} employer", f"{m.group('entity')} workplace"]),
        (r"^Which award (?P<entity>.+?) (?:received|earned|won|got)\?$", lambda m: [f"{m.group('entity')} award", f"awards of {m.group('entity')}"]),
        (r"^Who is (?P<entity>.+?)'s (?P<relation>father|mother|spouse|husband|wife|child|son|daughter|father-in-law|mother-in-law|uncle|aunt|stepmother|stepfather)\?$", lambda m: [f"{m.group('entity')} {m.group('relation')}"]),
        (r"^Who is the (?P<relation>father|mother|spouse|husband|wife|child|son|daughter|father-in-law|mother-in-law|uncle|aunt|stepmother|stepfather|maternal grandmother|maternal grandfather|paternal grandmother|paternal grandfather) of (?P<entity>.+?)\?$", lambda m: [f"{m.group('entity')} {m.group('relation')}"]),
    ]

    for pattern, builder in patterns:
        match = re.match(pattern, normalized_query, flags=re.IGNORECASE)
        if not match:
            continue
        for variant in builder(match):
            _append_variant(variants, variant)
        break

    possessive_match = re.search(r"(?P<entity>.+?)'s (?P<relation>.+)", normalized_query)
    if possessive_match:
        _append_variant(variants, f"{possessive_match.group('entity')} {possessive_match.group('relation')}")
        _append_variant(variants, f"{possessive_match.group('relation')} of {possessive_match.group('entity')}")

    return variants[:max_variants]


def _dedupe_results(results: List[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen = set()
    for result in results:
        if isinstance(result, dict):
            key = result.get('doc_id') or result.get('id') or result.get('text') or result.get('contents') or str(result)
        else:
            key = str(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def search_by_http(query: str, host: str = 'localhost', port: int = 8090) -> List[Dict]:
    url = f"http://{host}:{port}"
    response = requests.post(url, json={'query': query}, timeout=20)

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get a response. Status code: {response.status_code}")
        return []


def search_by_graph_api(query: str, url: str) -> List[Dict]:
    try:
        response = requests.post(url, json={'query': query}, headers={"Content-Type": "application/json"}, timeout=20)
        if response.status_code == 200:
            return _normalize_graph_api_results(response.json())
        else:
            logger.error(f"Failed to get a response from graph API. Status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error calling graph API: {e}")
        return []


def search_with_variants(
        query: str,
        graph_api_url: Optional[str] = None,
        host: str = 'localhost',
        port: int = 8090,
        max_variants: int = 4,
        per_query_limit: int = 5,
) -> List[Dict]:
    queries = build_search_queries(query, max_variants=max_variants)
    merged_results: List[Dict] = []

    search_fn: Callable[..., List[Dict]]
    search_kwargs: Dict[str, object]
    if graph_api_url:
        search_fn = search_by_graph_api
        search_kwargs = {'url': graph_api_url}
    else:
        search_fn = search_by_http
        search_kwargs = {'host': host, 'port': port}

    for candidate_query in queries:
        try:
            results = search_fn(query=candidate_query, **search_kwargs)
        except TypeError:
            logger.exception(f"Search invocation failed for query variant: {candidate_query}")
            continue
        if per_query_limit > 0:
            results = results[:per_query_limit]
        merged_results.extend(results)

    return _dedupe_results(merged_results)
