import os

# Disable tokenizer parallelism to avoid "Already borrowed" errors in multithreaded environment.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast

# Add src to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from agent import CoRagAgent, RagPath
from config import Arguments
from data_utils import format_documents_for_final_answer, load_corpus
from logger_config import logger
from search.search_utils import search_by_graph_api, search_by_http
from utils import AtomicCounter
from vllm_client import VllmClient, get_vllm_model_id


# Suppress verbose dependency logs.
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ScriptArguments:
    eval_file: str = field(default="./data/musique_hard.json", metadata={"help": "Path to input JSON file"})
    save_file: str = field(
        default="./eval/musique_hard_4B_v2data-2500_decomp-only_step3.json",
        metadata={"help": "Path to output JSON file"},
    )
    calc_recall: bool = field(default=True, metadata={"help": "Calculate retrieval recall"})
    enable_naive_retrieval: bool = field(
        default=True,
        metadata={"help": "Enable naive retrieval baseline comparison"},
    )


@dataclass
class EvalRuntime:
    args: Arguments
    corpus: Optional[Dataset]
    tokenizer: PreTrainedTokenizerFast
    tokenizer_lock: threading.Lock
    corag_agent: CoRagAgent
    processed_cnt: AtomicCounter
    total_cnt: int


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation/articles/extra spaces."""
    normalized = text.lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def split_sentences(text: str) -> List[str]:
    """Split text by sentence-ending punctuation followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def check_hit(retrieved_docs: List[str], golden_facts: List[str]) -> Tuple[int, int]:
    """
    Compute soft-inclusion hit count.

    Hit conditions:
    1) golden fact is substring of retrieved chunk;
    2) retrieved chunk is substring of golden fact and retrieved length > 0.5 * golden length.
    """
    if not golden_facts:
        return 0, 0

    hits = 0
    norm_gold = [normalize_text(g) for g in golden_facts]
    norm_retr = [normalize_text(r) for r in retrieved_docs]

    for golden_fact in norm_gold:
        if not golden_fact:
            continue
        is_hit = False
        for retrieved_doc in norm_retr:
            if not retrieved_doc:
                continue
            if golden_fact in retrieved_doc:
                is_hit = True
                break
            if retrieved_doc in golden_fact and len(retrieved_doc) > 0.5 * len(golden_fact):
                is_hit = True
                break
        if is_hit:
            hits += 1

    return hits, len(golden_facts)


def get_golden_facts(item: Dict[str, Any]) -> List[str]:
    """
    Extract golden facts from supported dataset structures.

    Supported formats:
    1) MuSiQue: item['paragraphs'][*]['is_supporting'] with paragraph_text split to sentences.
    2) HotpotQA: item['context'] + item['supporting_facts'] with sentence index lookup.
    """
    if "paragraphs" in item:
        facts: List[str] = []
        for paragraph in item["paragraphs"]:
            if paragraph.get("is_supporting"):
                text = paragraph.get("paragraph_text", "")
                if text:
                    facts.extend(split_sentences(text))
        return facts

    context = item.get("context", [])
    supporting_facts = item.get("supporting_facts", [])
    ctx_map = {ctx[0]: ctx[1] for ctx in context}

    facts = []
    for title, sent_idx in supporting_facts:
        if title in ctx_map:
            sentences = ctx_map[title]
            if 0 <= sent_idx < len(sentences):
                facts.append(sentences[sent_idx])
    return facts


def _resolve_model_id(model_name: Optional[str], api_base: str, api_key: str, log_prefix: str) -> str:
    if model_name:
        return model_name
    logger.info(f"Auto-detecting {log_prefix} model from {api_base}...")
    return get_vllm_model_id(api_base=api_base, api_key=api_key)


def _build_vllm_client(model_name: str, api_base: str, api_key: str) -> VllmClient:
    return VllmClient(model=model_name, api_base=api_base, api_key=api_key)


def _init_runtime(args: Arguments) -> EvalRuntime:
    logger.info("Initializing VLLM Client...")
    model_id = _resolve_model_id(args.vllm_model, args.vllm_api_base, args.vllm_api_key, "base")
    vllm_client = _build_vllm_client(model_id, args.vllm_api_base, args.vllm_api_key)

    final_vllm_client: Optional[VllmClient] = None
    if args.final_answer_model or args.final_answer_api_base:
        final_api_base = args.final_answer_api_base if args.final_answer_api_base else args.vllm_api_base
        final_api_key = args.final_answer_api_key if args.final_answer_api_key else args.vllm_api_key
        final_model_id = _resolve_model_id(args.final_answer_model, final_api_base, final_api_key, "final answer")
        logger.info(f"Initializing Final Answer VLLM Client ({final_model_id})...")
        final_vllm_client = _build_vllm_client(final_model_id, final_api_base, final_api_key)

    sub_answer_vllm_client: Optional[VllmClient] = None
    if args.sub_answer_model or args.sub_answer_api_base:
        sub_api_base = args.sub_answer_api_base if args.sub_answer_api_base else args.vllm_api_base
        sub_api_key = args.sub_answer_api_key if args.sub_answer_api_key else args.vllm_api_key
        sub_model_id = _resolve_model_id(args.sub_answer_model, sub_api_base, sub_api_key, "sub-answer")
        logger.info(f"Initializing Sub-Answer VLLM Client ({sub_model_id})...")
        sub_answer_vllm_client = _build_vllm_client(sub_model_id, sub_api_base, sub_api_key)

    logger.info("Loading Corpus...")
    corpus = load_corpus(args.corpus_file) if args.corpus_file else None
    if corpus is None and not args.corpus_file:
        logger.info("No corpus file provided. Skipping corpus loading.")

    logger.info("Initializing Agent...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else model_id
    try:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        logger.error(f"Failed to load tokenizer from '{tokenizer_name}'.")
        raise exc

    corag_agent = CoRagAgent(
        vllm_client=vllm_client,
        corpus=corpus,
        graph_api_url=args.graph_api_url,
        tokenizer=tokenizer,
        final_vllm_client=final_vllm_client,
        sub_answer_vllm_client=sub_answer_vllm_client,
    )

    return EvalRuntime(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        tokenizer_lock=corag_agent.lock,
        corag_agent=corag_agent,
        processed_cnt=AtomicCounter(),
        total_cnt=0,
    )


def _extract_question(item: Dict[str, Any]) -> str:
    question = item.get("question", "")
    if question:
        return question
    for key in item:
        if "question" in key.lower():
            return item[key]
    return ""


def _sample_path(runtime: EvalRuntime, question: str, task_desc: str) -> RagPath:
    args = runtime.args
    agent = runtime.corag_agent

    if args.decode_strategy == "greedy" or args.max_path_length < 1:
        return agent.sample_path(
            query=question,
            task_desc=task_desc,
            max_path_length=args.max_path_length,
            temperature=0.0,
            max_tokens=64,
        )
    if args.decode_strategy == "tree_search":
        return agent.tree_search(
            query=question,
            task_desc=task_desc,
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature,
            max_tokens=64,
        )
    if args.decode_strategy == "best_of_n":
        return agent.best_of_n(
            query=question,
            task_desc=task_desc,
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature,
            n=args.best_n,
            max_tokens=64,
        )
    # Keep behavior close to original script: unsupported strategy returns None and fails downstream.
    return None


def _collect_unique_documents(path: RagPath) -> List[Any]:
    all_documents: List[Any] = []
    if path.past_documents:
        for docs in path.past_documents:
            all_documents.extend(docs)

    seen = set()
    return [doc for doc in all_documents if not (doc in seen or seen.add(doc))]


def _format_final_documents(runtime: EvalRuntime, unique_documents: List[Any]) -> Any:
    with runtime.tokenizer_lock:
        return format_documents_for_final_answer(
            args=runtime.args,
            tokenizer=runtime.tokenizer,
            corpus=runtime.corpus,
            documents=unique_documents,
            lock=None,
        )


def _run_naive_retrieval(runtime: EvalRuntime, question: str) -> List[str]:
    args = runtime.args
    corpus = runtime.corpus
    naive_docs: List[str] = []

    if args.graph_api_url:
        retriever_results = search_by_graph_api(query=question, url=args.graph_api_url)
        for result in retriever_results:
            if isinstance(result, str):
                naive_docs.append(result)
            elif isinstance(result, dict):
                content = result.get("contents") or result.get("content") or result.get("text") or str(result)
                naive_docs.append(content)
    else:
        retriever_results = search_by_http(query=question)
        if corpus:
            from data_utils import format_input_context

            doc_ids = [result["doc_id"] for result in retriever_results]
            naive_docs = [format_input_context(corpus[int(doc_id)]) for doc_id in doc_ids]

    return naive_docs[: args.num_contexts]


def _process_item(item: Dict[str, Any], runtime: EvalRuntime) -> Optional[Dict[str, Any]]:
    args = runtime.args
    question = _extract_question(item)
    ground_truth = item.get("answer", "")
    if not question:
        logger.warning(f"Skipping item without question: {item}")
        return None

    task_desc = "answer multi-hop questions"
    start_time = time.time()

    path = _sample_path(runtime, question, task_desc)
    path_gen_time = time.time() - start_time

    unique_documents = _collect_unique_documents(path)
    documents = _format_final_documents(runtime, unique_documents)

    corag_recall_info: Dict[str, Any] = {}
    naive_recall_info: Dict[str, Any] = {}

    if args.calc_recall:
        golden_facts = get_golden_facts(item)
        c_hits, c_total = check_hit(unique_documents, golden_facts)
        corag_recall_info = {
            "hits": c_hits,
            "total": c_total,
            "recall": c_hits / c_total if c_total > 0 else 0.0,
        }

        if args.enable_naive_retrieval:
            naive_docs = _run_naive_retrieval(runtime, question)
            n_hits, n_total = check_hit(naive_docs, golden_facts)
            naive_recall_info = {
                "hits": n_hits,
                "total": n_total,
                "recall": n_hits / n_total if n_total > 0 else 0.0,
                "retrieved_docs": naive_docs,
            }

    prediction = runtime.corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc=task_desc,
        documents=documents,
        max_message_length=args.max_len,
        temperature=0.0,
        max_tokens=128,
    )

    total_time = time.time() - start_time
    final_gen_time = total_time - path_gen_time

    runtime.processed_cnt.increment()
    if runtime.processed_cnt.value % 10 == 0:
        logger.info(f"Processed {runtime.processed_cnt.value} / {runtime.total_cnt}")

    result_item: Dict[str, Any] = {
        "question": question,
        "answer": prediction,
        "ground_truth": ground_truth,
        "reasoning_steps": [],
        "time": [0.0, 0.0, path_gen_time, final_gen_time],
        "corag_recall": corag_recall_info if args.calc_recall else None,
        "naive_recall": naive_recall_info if args.calc_recall and args.enable_naive_retrieval else None,
    }

    if path.past_subqueries:
        for subquery, subanswer in zip(path.past_subqueries, path.past_subanswers):
            result_item["reasoning_steps"].append({"subquery": subquery, "subanswer": subanswer})

    return result_item


def _summarize_results(processed_results: List[Dict[str, Any]], args: Arguments) -> Dict[str, Any]:
    total_q2t = 0.0
    total_ppr = 0.0
    total_reranker = 0.0
    total_llm_call = 0.0

    total_corag_hits = 0
    total_naive_hits = 0
    total_gold_chunks = 0

    for result in processed_results:
        t = result["time"]
        total_q2t += t[0]
        total_ppr += t[1]
        total_reranker += t[2]
        total_llm_call += t[3]

        if args.calc_recall and result.get("corag_recall"):
            total_corag_hits += result["corag_recall"]["hits"]
            total_gold_chunks += result["corag_recall"]["total"]

        if args.calc_recall and args.enable_naive_retrieval and result.get("naive_recall"):
            total_naive_hits += result["naive_recall"]["hits"]

    num_samples = len(processed_results)
    summary = {
        "type": "Summary",
        "total_samples": num_samples,
        "avg_q2t_time": total_q2t / num_samples if num_samples > 0 else 0,
        "avg_ppr_time": total_ppr / num_samples if num_samples > 0 else 0,
        "avg_reranker_time": total_reranker / num_samples if num_samples > 0 else 0,
        "avg_llm_call_time": total_llm_call / num_samples if num_samples > 0 else 0,
    }

    if args.calc_recall:
        summary["corag_micro_recall"] = total_corag_hits / total_gold_chunks if total_gold_chunks > 0 else 0.0
        if args.enable_naive_retrieval:
            summary["naive_micro_recall"] = total_naive_hits / total_gold_chunks if total_gold_chunks > 0 else 0.0

    return summary


def _save_results(results: List[Dict[str, Any]], save_file: str) -> None:
    logger.info(f"Saving results to {save_file}...")
    save_dir = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


def run_custom_eval(args: Arguments) -> None:
    runtime = _init_runtime(args)

    if args.max_path_length < 1:
        args.decode_strategy = "greedy"

    logger.info(f"Loading custom dataset from {args.eval_file}...")
    with open(args.eval_file, "r", encoding="utf-8") as file:
        data_items = json.load(file)

    if args.dry_run:
        logger.info("Dry run enabled: processing only first 2 items.")
        data_items = data_items[:2]

    runtime.total_cnt = len(data_items)

    logger.info(f"Processing {runtime.total_cnt} items with {args.num_threads} threads...")

    results_map: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_index = {
            executor.submit(_process_item, item, runtime): index for index, item in enumerate(data_items)
        }
        for future in tqdm(as_completed(future_to_index), total=runtime.total_cnt, desc="Processing"):
            index = future_to_index[future]
            try:
                result = future.result()
                if result:
                    results_map[index] = result
            except Exception as exc:
                logger.error(f"Error processing item at index {index}: {exc}")
                import traceback

                traceback.print_exc()

    processed_results = [results_map[i] for i in range(len(data_items)) if i in results_map]
    summary = _summarize_results(processed_results, args)
    processed_results.insert(0, summary)

    _save_results(processed_results, args.save_file)
    logger.info("Done!")


def parse_arguments() -> Arguments:
    parser = HfArgumentParser((Arguments,))
    _, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")

    parser = HfArgumentParser((Arguments, ScriptArguments))
    args, script_args = parser.parse_args_into_dataclasses()

    args.eval_file = script_args.eval_file
    args.save_file = script_args.save_file
    args.calc_recall = script_args.calc_recall
    args.enable_naive_retrieval = script_args.enable_naive_retrieval
    return args


if __name__ == "__main__":
    runtime_args = parse_arguments()
    if not runtime_args.eval_file or not runtime_args.save_file:
        logger.error("Please provide --eval_file and --save_file")
        sys.exit(1)

    run_custom_eval(runtime_args)
