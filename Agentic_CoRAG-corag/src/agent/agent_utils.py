from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RagPath:

    query: str
    past_subqueries: Optional[List[str]]
    past_subanswers: Optional[List[str]]
    past_doc_ids: Optional[List[List[str]]]
    past_thoughts: Optional[List[str]] = None
    past_documents: Optional[List[List[str]]] = None
    past_retrieval_stats: Optional[List[Dict[str, Any]]] = None
