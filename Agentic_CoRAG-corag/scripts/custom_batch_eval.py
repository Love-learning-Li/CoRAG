import sys
import os
# Disable tokenizer parallelism to avoid "Already borrowed" errors in multithreaded environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import logging
import threading
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_path)

from transformers import HfArgumentParser, AutoTokenizer, PreTrainedTokenizerFast
from datasets import Dataset
from config import Arguments
from data_utils import load_corpus, format_documents_for_final_answer
from vllm_client import VllmClient, get_vllm_model_id
from agent import CoRagAgent, RagPath
from search.search_utils import extract_retrieved_documents, search_by_http, search_by_graph_api, search_with_variants
from utils import AtomicCounter
import re
from logger_config import logger

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def load_tokenizer_safely(tokenizer_name: str) -> PreTrainedTokenizerFast:
    """Load tokenizer with better offline handling and clearer diagnostics."""
    # If a local directory is provided, force local load to avoid unnecessary hub access.
    if os.path.isdir(tokenizer_name):
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    try:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as first_error:
        # Retry in local-only mode in case files are already cached.
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        except Exception:
            raise RuntimeError(
                "Failed to load tokenizer. If this environment cannot access huggingface.co, "
                "please set --tokenizer_name to a local tokenizer directory, "
                "or pre-download/cache the model files. "
                f"Current tokenizer_name: {tokenizer_name}"
            ) from first_error

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def check_hit(retrieved_docs: List[str], golden_facts: List[str]) -> Tuple[int, int]:
    """
    Calculate hit count based on Soft Inclusion.
    Hit if:
    1. Golden chunk is substring of Retrieved chunk (Golden in Retrieved)
    2. Retrieved chunk is substring of Golden chunk AND len(Retrieved) > 0.5 * len(Golden)
    """
    hits = 0
    if not golden_facts:
        return 0, 0
    
    # Pre-normalize
    norm_gold = [normalize_text(g) for g in golden_facts]
    norm_retr = [normalize_text(r) for r in retrieved_docs]
    
    # To count hits unique per golden fact (i.e., did we retrieve this fact?)
    # We iterate over golden facts and check if ANY retrieved doc covers it.
    for golden_fact in norm_gold:
        is_hit = False
        if not golden_fact:
            continue
        for retrieved_doc in norm_retr:
            if not retrieved_doc:
                continue
            
            # Condition 1: Golden in Retrieved
            if golden_fact in retrieved_doc:
                is_hit = True
                break
            
            # Condition 2: Retrieved in Golden (Significant fragment)
            if retrieved_doc in golden_fact and len(retrieved_doc) > 0.5 * len(golden_fact):
                is_hit = True
                break
        
        if is_hit:
            hits += 1
            
    return hits, len(golden_facts)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation (.?!) followed by whitespace."""
    # Split after .?! followed by whitespace
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]

def get_golden_facts(item: Dict[str, Any]) -> List[str]:
    """
    Parse item to extract golden facts.
    Supports:
    1. MuSiQue: "paragraphs" list with "is_supporting" flag -> split into sentences.
    2. HotpotQA: "context" and "supporting_facts" -> extract specific sentences.
    """
    # 1. MuSiQue Check
    if 'paragraphs' in item:
        facts = []
        for p in item['paragraphs']:
            if p.get('is_supporting'):
                text = p.get('paragraph_text', '')
                if text:
                    sentences = split_sentences(text)
                    facts.extend(sentences)
        return facts

    # 2. HotpotQA Logic (Fallback)
    context = item.get('context', [])
    supporting_facts = item.get('supporting_facts', [])
    
    facts = []
    
    # Create a map for quick access: title -> list of sentences
    ctx_map = {c[0]: c[1] for c in context}
    
    for title, sent_idx in supporting_facts:
        if title in ctx_map:
            sentences = ctx_map[title]
            if 0 <= sent_idx < len(sentences):
                facts.append(sentences[sent_idx])
    
    return facts

def run_custom_eval(args: Arguments):
    # Initialize components
    logger.info("Initializing VLLM Client...")
    if args.vllm_model:
        model_id = args.vllm_model
    else:
        model_id = get_vllm_model_id(api_base=args.vllm_api_base, api_key=args.vllm_api_key)
    
    vllm_client: VllmClient = VllmClient(
        model=model_id, api_base=args.vllm_api_base, api_key=args.vllm_api_key
        )
    
    final_vllm_client: VllmClient = None
    if args.final_answer_model or args.final_answer_api_base:
        final_api_base = args.final_answer_api_base if args.final_answer_api_base else args.vllm_api_base
        final_api_key = args.final_answer_api_key if args.final_answer_api_key else args.vllm_api_key
        
        if args.final_answer_model:
            final_model_id = args.final_answer_model
        else:
             logger.info(f"Auto-detecting Final Answer Model from {final_api_base}...")
             final_model_id = get_vllm_model_id(api_base=final_api_base, api_key=final_api_key)

        logger.info(f"Initializing Final Answer VLLM Client ({final_model_id})...")
        final_vllm_client = VllmClient(model=final_model_id, api_base=final_api_base, api_key=final_api_key)
    
    sub_answer_vllm_client: VllmClient = None
    if args.sub_answer_model or args.sub_answer_api_base:
        sub_api_base = args.sub_answer_api_base if args.sub_answer_api_base else args.vllm_api_base
        sub_api_key = args.sub_answer_api_key if args.sub_answer_api_key else args.vllm_api_key
        
        if args.sub_answer_model:
            sub_model_id = args.sub_answer_model
        else:
            logger.info(f"Auto-detecting Sub-Answer Model from {sub_api_base}...")
            sub_model_id = get_vllm_model_id(api_base=sub_api_base, api_key=sub_api_key)
        
        logger.info(f"Initializing Sub-Answer VLLM Client ({sub_model_id})...")
        sub_answer_vllm_client = VllmClient(model=sub_model_id, api_base=sub_api_base, api_key=sub_api_key)
    
    logger.info("Loading Corpus...")
    corpus = None
    if args.corpus_file:
        corpus: Dataset = load_corpus(args.corpus_file)
    else:
        logger.info("No corpus file provided. Skipping corpus loading.")
    
    logger.info("Initializing Agent...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else model_id
    try:
        tokenizer: PreTrainedTokenizerFast = load_tokenizer_safely(tokenizer_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from '{tokenizer_name}'.")
        raise e
        
    corag_agent: CoRagAgent = CoRagAgent(
        vllm_client=vllm_client, 
        corpus=corpus, 
        graph_api_url=args.graph_api_url, 
        tokenizer=tokenizer,
        final_vllm_client=final_vllm_client,
        sub_answer_vllm_client=sub_answer_vllm_client,
        retrieval_max_variants=args.retrieval_max_variants,
        retrieval_per_query_limit=args.retrieval_per_query_limit,
        retrieval_service_top_k=args.retrieval_service_top_k,
        enable_deterministic_planner=args.enable_deterministic_planner,
        enable_generic_deterministic_rules=args.enable_generic_deterministic_rules,
        enable_dataset_specific_rules=args.enable_dataset_specific_rules,
        dataset_rule_profile=args.dataset_rule_profile,
    )
    # Use the same lock as the agent to ensure thread safety for tokenizer access
    tokenizer_lock: threading.Lock = corag_agent.lock

    # Pre-flight: verify the retrieval service is reachable before starting the eval.
    # A failure here saves hours of timeout-wasted wall time.
    logger.info("Pre-flight: checking retrieval service connectivity...")
    _test_query = "test connectivity"
    _t0 = time.time()
    if args.graph_api_url:
        _test_results = search_by_graph_api(query=_test_query, url=args.graph_api_url, top_k=args.retrieval_service_top_k)
    else:
        _test_results = search_by_http(query=_test_query, top_k=args.retrieval_service_top_k)
    _elapsed = time.time() - _t0
    # For graph/PPR retrieval a single cold-start call legitimately takes 30-120 s.
    # Flag as a problem only when it returns no results (hard failure), not just slowness.
    _slow_threshold = 60.0 if args.graph_api_url else 4.0
    if not _test_results:
        logger.warning(
            f"RETRIEVAL SERVICE ISSUE DETECTED: test query took {_elapsed:.1f}s and returned "
            f"{len(_test_results)} result(s).  All searches during this eval will likely return "
            "empty, causing recall=0 and answers driven entirely by LLM parametric memory.  "
            "Please ensure the retrieval service is running and accessible before re-running."
        )
    elif _elapsed > _slow_threshold:
        logger.warning(
            f"Retrieval service responded but is SLOW ({_elapsed:.1f}s, {len(_test_results)} result(s)). "
            f"Under concurrent load the gc_service may not keep up. Consider using --num_threads 1 or 2."
        )
    else:
        logger.info(f"Retrieval service OK ({_elapsed:.2f}s, {len(_test_results)} result(s) for test query).")
    # Log the raw format of the first result so we can verify the field names
    # that the retrieval service actually returns (for format-mismatch diagnosis).
    if _test_results:
        first = _test_results[0]
        logger.info(
            f"[FORMAT PROBE] type={type(first).__name__}, "
            f"value_preview={repr(first)[:300]}"
        )
        if isinstance(first, dict):
            logger.info(f"[FORMAT PROBE] dict keys: {list(first.keys())}")

    if args.max_path_length < 1:
        args.decode_strategy = 'greedy'

    # Safety check: graph PPR retrieval is CPU-bound and serialised by the GIL on the
    # server side.  Running many client threads hammers a single gc_service worker and
    # causes cascading read timeouts (each queued request waits ~PPR_time × queue_depth).
    # With PPR taking 30-120 s per call:
    #   - 1 thread:  queue_depth=1  → wait ~30-120 s  (safe with 240 s timeout)
    #   - 2 threads: queue_depth=2  → wait ~60-240 s  (borderline with 240 s timeout)
    #   - 3+ threads: queue_depth=3 → wait ~90-360 s  (likely to exceed 240 s timeout)
    # Strong recommendation: --num_threads 1
    if args.graph_api_url and args.num_threads > 2:
        logger.warning(
            f"num_threads={args.num_threads} is likely too high for graph/PPR retrieval. "
            "The gc_service PPR operation is CPU-bound and effectively single-threaded "
            "due to the Python GIL. Each extra concurrent thread adds ~30-120 s of "
            "queue wait, quickly exceeding the read timeout even with retry. "
            "Strongly recommended: --num_threads 1  (safe) or --num_threads 2 (borderline)"
        )

    # Load custom dataset
    logger.info(f"Loading custom dataset from {args.eval_file}...")
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)

    if args.dry_run:
        logger.info("Dry run enabled: processing only first 2 items.")
        data_items = data_items[:2]

    processed_cnt = AtomicCounter()
    total_cnt = len(data_items)
    failed_cnt = AtomicCounter()
    
    def process_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        question = item.get('question', '')
        ground_truth = item.get('answer', '')
        # Fallback if keys are different
        if not question:
            # Try finding a key that looks like a question
            for k in item.keys():
                if 'question' in k.lower():
                    question = item[k]
                    break
        
        if not question:
            logger.warning(f"Skipping item without question: {item}")
            return None

        task_desc = "answer multi-hop questions" # Default
        
        start_time = time.time()
        
        # 1. Path Generation
        path: RagPath = None
        if args.decode_strategy == 'greedy' or args.max_path_length < 1:
            path = corag_agent.sample_path(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=0., max_tokens=64
            )
        elif args.decode_strategy == 'tree_search':
            path = corag_agent.tree_search(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=args.sample_temperature, max_tokens=64
            )
        elif args.decode_strategy == 'best_of_n':
            path = corag_agent.best_of_n(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=args.sample_temperature,
                n = args.best_n,
                max_tokens=64
            )
        path_gen_time = time.time() - start_time
        
        # 2. Document Formatting
        all_documents = []
        if path.past_documents:
            for docs in path.past_documents:
                all_documents.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_documents = [x for x in all_documents if not (x in seen or seen.add(x))]

        # Use lock when modifying Documents via tokenizer (it modifies internal state)
        with tokenizer_lock:
            documents = format_documents_for_final_answer(
                args=args,
                tokenizer=tokenizer,
                corpus=corpus,
                documents=unique_documents,
                lock=None # The lock is already held outside this call
            )

        # 4. Recall Evaluation
        corag_recall_info = {}
        naive_recall_info = {}
        retrieval_debug = {
            "corag_steps": getattr(path, "past_retrieval_stats", []) or [],
        }
        
        if args.calc_recall:
            # A. Get Golden Facts
            golden_facts = get_golden_facts(item)
            
            # B. CoRAG Recall
            # Use the same document budget that reaches final-answer generation.
            c_hits, c_total = check_hit(documents, golden_facts)
            corag_recall_info = {
                "hits": c_hits,
                "total": c_total,
                "recall": c_hits / c_total if c_total > 0 else 0.0,
                "retrieved_docs": documents,
            }
            
            # C. Naive Retrieval Recall
            # Uses search_with_variants (same variant expansion as CoRAG) for a fair comparison.
            if args.enable_naive_retrieval:
                naive_raw = search_with_variants(
                    query=question,
                    graph_api_url=args.graph_api_url if args.graph_api_url else None,
                    max_variants=args.retrieval_max_variants,
                    per_query_limit=args.retrieval_per_query_limit,
                    service_top_k=args.retrieval_service_top_k,
                )
                _, naive_docs, naive_format_issue_count = extract_retrieved_documents(
                    naive_raw,
                    reverse_order=False,
                    limit=args.num_contexts,
                    corpus=corpus,
                )

                n_hits, n_total = check_hit(naive_docs, golden_facts)
                naive_recall_info = {
                    "hits": n_hits,
                    "total": n_total,
                    "recall": n_hits / n_total if n_total > 0 else 0.0,
                    "retrieved_docs": naive_docs
                }
                retrieval_debug["naive"] = {
                    "raw_result_count": len(naive_raw),
                    "usable_result_count": len(naive_docs),
                    "format_issue_count": naive_format_issue_count,
                }

        # 3. Final Answer Generation
        # Time this stage separately so the recall-evaluation overhead does not
        # contaminate the reported final_generation latency.
        final_gen_start = time.time()
        prediction: str = corag_agent.generate_final_answer(
            corag_sample=path,
            task_desc=task_desc,
            documents=documents,
            max_message_length=args.max_len,
            temperature=0., max_tokens=128
        )
        final_gen_time = time.time() - final_gen_start

        # Logging
        processed_cnt.increment()
        if processed_cnt.value % 10 == 0:
            logger.info(f"Processed {processed_cnt.value} / {total_cnt}")

        # Construct result dict
        # Custom metrics if available (Q2T, PPR not captured here)
        time_breakdown = [0.0, 0.0, path_gen_time, final_gen_time]

        result_item = {
            "question": question,
            "answer": prediction,
            "ground_truth": ground_truth,
            "reasoning_steps": [],
            "time": time_breakdown,
            "corag_recall": corag_recall_info if args.calc_recall else None,
            "naive_recall": naive_recall_info if args.calc_recall and args.enable_naive_retrieval else None,
            "retrieval_debug": retrieval_debug,
        }

        # Add reasoning steps
        if path.past_subqueries:
            for sq, sa in zip(path.past_subqueries, path.past_subanswers):
                result_item["reasoning_steps"].append({
                    "subquery": sq,
                    "subanswer": sa
                })
        
        return result_item

    # Use ThreadPoolExecutor for parallel processing
    # Note: Adjust num_threads in args based on system capabilities
    logger.info(f"Processing {total_cnt} items with {args.num_threads} threads...")
    
    processed_results = []
    
    # Use a dictionary to store results by index to preserve order
    results_map = {}
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks and map future to index
        future_to_index = {executor.submit(process_item, item): i for i, item in enumerate(data_items)}
        
        # Use as_completed so tqdm updates as soon as ANY thread finishes
        for future in tqdm(as_completed(future_to_index), total=total_cnt, desc="Processing"):
            index = future_to_index[future]
            try:
                res = future.result()
                if res:
                    results_map[index] = res
            except Exception as e:
                failed_cnt.increment()
                logger.error(f"Error processing item at index {index}: {e}")
                import traceback
                traceback.print_exc()

    # Reconstruct results in original order
    processed_results = []
    for i in range(len(data_items)):
        if i in results_map:
            processed_results.append(results_map[i])

    if not processed_results:
        raise RuntimeError(
            f"All {total_cnt} items failed during evaluation. "
            f"Please check vLLM service connectivity, API base configuration, and retriever endpoints."
        )

    # Calculate average stats
    total_q2t = 0
    total_ppr = 0
    total_reranker = 0
    total_llm_call = 0
    
    # Recall aggregators
    total_corag_hits = 0
    total_naive_hits = 0
    total_gold_chunks = 0
    
    for res in processed_results:
        t = res["time"]
        total_q2t += t[0]
        total_ppr += t[1]
        total_reranker += t[2]
        total_llm_call += t[3]
        
        if args.calc_recall and res.get("corag_recall"):
            total_corag_hits += res["corag_recall"]["hits"]
            total_gold_chunks += res["corag_recall"]["total"] # Summing total gold chunks across all questions?
            # Or should we average per-question recall?
            # User said: "命中所有题目的golden chunks/总共golden chunks取平均" -> Micro Recall
            
        if args.calc_recall and args.enable_naive_retrieval and res.get("naive_recall"):
            total_naive_hits += res["naive_recall"]["hits"]
            # total_gold_chunks is same for both
            
    num_samples = len(processed_results)
    avg_summary = {
        "type": "Summary",
        "total_samples": num_samples,
        "failed_samples": failed_cnt.value,
        "avg_q2t_time": total_q2t / num_samples if num_samples > 0 else 0,
        "avg_ppr_time": total_ppr / num_samples if num_samples > 0 else 0,
        "avg_reranker_time": total_reranker / num_samples if num_samples > 0 else 0,
        "avg_llm_call_time": total_llm_call / num_samples if num_samples > 0 else 0,
    }
    
    if args.calc_recall:
        avg_summary["corag_micro_recall"] = total_corag_hits / total_gold_chunks if total_gold_chunks > 0 else 0.0
        if args.enable_naive_retrieval:
             avg_summary["naive_micro_recall"] = total_naive_hits / total_gold_chunks if total_gold_chunks > 0 else 0.0
    
    processed_results.insert(0, avg_summary)
    
    # Save results
    logger.info(f"Saving results to {args.save_file}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    
    with open(args.save_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)
        
    logger.info("Done!")

if __name__ == "__main__":
    # Define script-specific arguments separate from the main configuration
    from dataclasses import dataclass, field
    
    @dataclass
    class ScriptArguments:
        eval_file: str = field(default='./data/musique_hard.json', metadata={"help": "Path to input JSON file"})
        save_file: str = field(default='./eval/musique_hard_4B_v2data-2500_decomp-only_step3.json', metadata={"help": "Path to output JSON file"})
        calc_recall: bool = field(default=True, metadata={"help": "Calculate retrieval recall"})
        enable_naive_retrieval: bool = field(default=True, metadata={"help": "Enable naive retrieval baseline comparison"})
        
    parser = HfArgumentParser((Arguments, ScriptArguments))
    args, script_args = parser.parse_args_into_dataclasses()
    
    # Merge script arguments into the main arguments object
    args.eval_file = script_args.eval_file
    args.save_file = script_args.save_file
    args.calc_recall = script_args.calc_recall
    args.enable_naive_retrieval = script_args.enable_naive_retrieval
    
    if not args.eval_file or not args.save_file:
        logger.error("Please provide --eval_file and --save_file")
        sys.exit(1)

    run_custom_eval(args)
