import argparse
import json
import math
import re
import string
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CANDIDATES = [
    "2wikimultihopqa_hard_full_eval_out_3_13.json",
    "2wikimultihopqa_hard_full_eval_output_3_13_17_43.json",
    "2wikimultihopqa_hard_full_eval_out.json",
]


def resolve_default_eval_file() -> Path:
    eval_dir = REPO_ROOT / "eval"
    for filename in DEFAULT_EVAL_CANDIDATES:
        candidate = eval_dir / filename
        if candidate.exists():
            return candidate
    return eval_dir / DEFAULT_EVAL_CANDIDATES[0]


DEFAULT_EVAL_FILE = resolve_default_eval_file()


def normalize_squad(text: str) -> str:
    if text is None:
        return ""

    def lower(value: str) -> str:
        return value.lower()

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def extract_model_answer(text: Optional[str]) -> str:
    if text is None:
        return ""

    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for label in ("Final Answer:", "SubAnswer:", "SubQuery:"):
        pattern = re.compile(
            rf'{re.escape(label)}\s*(.*?)(?=(?:\n\s*(?:SubQuery|SubAnswer|Final Answer)\s*:)|$)',
            flags=re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.findall(cleaned)
        if matches:
            cleaned = matches[-1].strip()
            break

    cleaned = re.sub(r'^\s*(?:Final Answer|SubAnswer|SubQuery|Answer)\s*:\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = strip_wrapping_quotes(cleaned)
    lowered = cleaned.lower()
    if not cleaned or lowered == 'no relevant information found':
        return 'No relevant information found'
    if lowered in {'yes', 'no', 'insufficient information'}:
        return lowered
    return cleaned


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_squad(prediction) == normalize_squad(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_squad(prediction).split()
    gold_tokens = normalize_squad(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common: Dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, ground_truth: str) -> float:
    norm_pred = normalize_squad(prediction)
    norm_gt = normalize_squad(ground_truth)
    if not norm_pred or not norm_gt:
        return 0.0
    padded_pred = f" {norm_pred} "
    padded_gt = f" {norm_gt} "
    return float(padded_gt in padded_pred or padded_pred in padded_gt)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}

    return {
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "min": min(values),
    }


def summarize_times(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    path_times = [sample["time"][2] for sample in samples]
    final_times = [sample["time"][3] for sample in samples]
    total_times = [sample["time"][2] + sample["time"][3] for sample in samples]

    return {
        "path_generation": build_distribution(path_times),
        "final_generation": build_distribution(final_times),
        "end_to_end": build_distribution(total_times),
    }


def analyze_answer_quality(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    em_scores: List[float] = []
    f1_scores: List[float] = []
    contains_scores: List[float] = []
    yesno_total = 0
    yesno_correct = 0
    no_info_count = 0

    for sample in samples:
        prediction = extract_model_answer(sample.get("answer", ""))
        ground_truth = sample.get("ground_truth", "")

        em_scores.append(exact_match_score(prediction, ground_truth))
        f1_scores.append(token_f1_score(prediction, ground_truth))
        contains_scores.append(contains_match(prediction, ground_truth))

        if "no relevant information found" in prediction.lower():
            no_info_count += 1

        norm_gt = normalize_squad(ground_truth)
        norm_pred = normalize_squad(prediction)

        if norm_gt in {"yes", "no"}:
            yesno_total += 1
            if norm_gt == norm_pred:
                yesno_correct += 1

    return {
        "exact_match": mean(em_scores) if em_scores else 0.0,
        "token_f1": mean(f1_scores) if f1_scores else 0.0,
        "contains_match": mean(contains_scores) if contains_scores else 0.0,
        "yes_no_accuracy": yesno_correct / yesno_total if yesno_total else 0.0,
        "yes_no_total": yesno_total,
        "no_relevant_info_rate": no_info_count / len(samples) if samples else 0.0,
    }


def analyze_recall(samples: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    recalls = [sample[key]["recall"] for sample in samples if sample.get(key)]
    hits = sum(sample[key]["hits"] for sample in samples if sample.get(key))
    total = sum(sample[key]["total"] for sample in samples if sample.get(key))
    return {
        "micro_recall": hits / total if total else 0.0,
        "macro_recall": mean(recalls) if recalls else 0.0,
        "total_hits": hits,
        "total_gold": total,
    }


def compare_recall(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    corag_better = 0
    naive_better = 0
    tie = 0

    for sample in samples:
        corag = sample.get("corag_recall")
        naive = sample.get("naive_recall")
        if not corag or not naive:
            continue

        if corag["recall"] > naive["recall"]:
            corag_better += 1
        elif corag["recall"] < naive["recall"]:
            naive_better += 1
        else:
            tie += 1

    return {"corag_better": corag_better, "naive_better": naive_better, "tie": tie}


def analyze_reasoning(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_counts = [len(sample.get("reasoning_steps", [])) for sample in samples]
    return {
        "avg_steps": mean(step_counts) if step_counts else 0.0,
        "median_steps": median(step_counts) if step_counts else 0.0,
        "max_steps": max(step_counts) if step_counts else 0,
        "min_steps": min(step_counts) if step_counts else 0,
    }


def _question_category(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["same nationality", "same country", "same place", "share the same nationality", "share the same country"]):
        return "comparison"
    if any(token in lowered for token in ["father-in-law", "mother-in-law", "maternal ", "paternal ", "stepmother", "stepfather"]):
        return "complex_kinship"
    if any(token in lowered for token in ["director of film", "composer of film", "performer of song", "composer of song", "director of the film", "composer of the film"]):
        return "role_bridge"
    return "other"


def _is_no_info(text: str) -> bool:
    return "no relevant information found" in extract_model_answer(text).lower()


def _first_step_failed(sample: Dict[str, Any]) -> bool:
    steps = sample.get("reasoning_steps", [])
    return bool(steps) and _is_no_info(steps[0].get("subanswer", ""))


def _rewrites_after_first_failure(sample: Dict[str, Any]) -> bool:
    steps = sample.get("reasoning_steps", [])
    if len(steps) < 2 or not _first_step_failed(sample):
        return False
    first = normalize_squad(steps[0].get("subquery", ""))
    for step in steps[1:]:
        current = normalize_squad(step.get("subquery", ""))
        if current and current != first:
            return True
    return False


def _rewrite_salvaged(sample: Dict[str, Any]) -> bool:
    if not _rewrites_after_first_failure(sample):
        return False
    for step in sample.get("reasoning_steps", [])[1:]:
        if not _is_no_info(step.get("subanswer", "")):
            return True
    return False


def _relation_drift(sample: Dict[str, Any]) -> bool:
    question = sample.get("question", "").lower()
    step_queries = " ".join(step.get("subquery", "").lower() for step in sample.get("reasoning_steps", []))
    if "father-in-law" in question and "spouse" not in step_queries:
        return True
    if "maternal grandmother" in question and "mother" not in step_queries:
        return True
    if "paternal grandmother" in question and "father" not in step_queries:
        return True
    if "stepmother" in question and "father" not in step_queries:
        return True
    return False


def _comparison_one_side_missing(sample: Dict[str, Any]) -> bool:
    if _question_category(sample.get("question", "")) != "comparison":
        return False
    steps = sample.get("reasoning_steps", [])
    if not steps:
        return False
    informative_steps = [step for step in steps if not _is_no_info(step.get("subanswer", ""))]
    return len(informative_steps) == 1


def _count_format_issues(sample: Dict[str, Any], side: str) -> int:
    debug = sample.get("retrieval_debug", {})
    if side == "corag":
        return sum(int(step.get("format_issue_count", 0)) for step in debug.get("corag_steps", []) if isinstance(step, dict))
    side_info = debug.get("naive", {})
    if isinstance(side_info, dict):
        return int(side_info.get("format_issue_count", 0))
    return 0


def analyze_retrieval_failures(samples: List[Dict[str, Any]], top_failure_count: int = 8) -> Dict[str, Any]:
    total = len(samples)
    zero_corag = [
        sample for sample in samples
        if sample.get("corag_recall") is not None and sample["corag_recall"].get("recall", 0.0) == 0.0
    ]
    no_info = [sample for sample in samples if _is_no_info(sample.get("answer", ""))]
    yes_no_pool = [sample for sample in samples if normalize_squad(sample.get("ground_truth", "")) in {"yes", "no"}]
    yes_no_errors = [
        sample for sample in yes_no_pool
        if normalize_squad(extract_model_answer(sample.get("answer", ""))) != normalize_squad(sample.get("ground_truth", ""))
    ]

    first_step_failed = [sample for sample in samples if _first_step_failed(sample)]
    rewritten_after_failure = [sample for sample in samples if _rewrites_after_first_failure(sample)]
    rewrite_salvaged = [sample for sample in samples if _rewrite_salvaged(sample)]
    relation_drift = [sample for sample in samples if _relation_drift(sample)]
    comparison_one_side_missing = [sample for sample in samples if _comparison_one_side_missing(sample)]
    complex_kinship = [sample for sample in samples if _question_category(sample.get("question", "")) == "complex_kinship"]
    complex_kinship_fail = [
        sample for sample in complex_kinship
        if sample.get("corag_recall", {}).get("recall", 0.0) == 0.0
    ]
    role_bridge = [sample for sample in samples if _question_category(sample.get("question", "")) == "role_bridge"]
    role_bridge_fail = [
        sample for sample in role_bridge
        if sample.get("corag_recall", {}).get("recall", 0.0) == 0.0
    ]

    corag_format_issue_total = sum(_count_format_issues(sample, "corag") for sample in samples)
    naive_format_issue_total = sum(_count_format_issues(sample, "naive") for sample in samples)

    failure_examples = [
        {
            "question": sample.get("question", ""),
            "first_subquery": (sample.get("reasoning_steps") or [{}])[0].get("subquery", ""),
            "first_subanswer": (sample.get("reasoning_steps") or [{}])[0].get("subanswer", ""),
            "corag_recall": sample.get("corag_recall", {}).get("recall", 0.0),
            "final_answer": extract_model_answer(sample.get("answer", "")),
        }
        for sample in zero_corag[:top_failure_count]
    ]

    category_stats: Dict[str, Dict[str, float]] = {}
    for category in ("comparison", "complex_kinship", "role_bridge", "other"):
        bucket = [sample for sample in samples if _question_category(sample.get("question", "")) == category]
        if not bucket:
            continue
        zero_bucket = [sample for sample in bucket if sample.get("corag_recall", {}).get("recall", 0.0) == 0.0]
        category_stats[category] = {
            "count": len(bucket),
            "zero_recall_rate": len(zero_bucket) / len(bucket),
            "avg_corag_recall": mean(sample.get("corag_recall", {}).get("recall", 0.0) for sample in bucket),
        }

    return {
        "zero_corag_recall_rate": len(zero_corag) / total if total else 0.0,
        "no_info_rate": len(no_info) / total if total else 0.0,
        "yes_no_error_rate": len(yes_no_errors) / len(yes_no_pool) if yes_no_pool else 0.0,
        "first_step_failure_rate": len(first_step_failed) / total if total else 0.0,
        "rewrite_after_failure_rate": len(rewritten_after_failure) / len(first_step_failed) if first_step_failed else 0.0,
        "rewrite_salvage_rate": len(rewrite_salvaged) / len(rewritten_after_failure) if rewritten_after_failure else 0.0,
        "relation_drift_count": len(relation_drift),
        "comparison_one_side_missing_rate": len(comparison_one_side_missing) / len([sample for sample in samples if _question_category(sample.get("question", "")) == "comparison"]) if any(_question_category(sample.get("question", "")) == "comparison" for sample in samples) else 0.0,
        "complex_kinship_failure_rate": len(complex_kinship_fail) / len(complex_kinship) if complex_kinship else 0.0,
        "role_bridge_failure_rate": len(role_bridge_fail) / len(role_bridge) if role_bridge else 0.0,
        "corag_format_issue_total": corag_format_issue_total,
        "naive_format_issue_total": naive_format_issue_total,
        "category_stats": category_stats,
        "failure_examples": failure_examples,
    }


def get_slowest_samples(samples: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked = sorted(samples, key=lambda sample: sample["time"][2] + sample["time"][3], reverse=True)
    return [
        {
            "question": sample.get("question", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "prediction": sample.get("answer", ""),
            "path_generation_time": sample["time"][2],
            "final_generation_time": sample["time"][3],
            "total_time": sample["time"][2] + sample["time"][3],
            "reasoning_steps": len(sample.get("reasoning_steps", [])),
        }
        for sample in ranked[:top_k]
    ]


def load_eval_file(file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with file_path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)

    if not isinstance(content, list) or not content:
        raise ValueError("Evaluation file must be a non-empty list.")

    summary = content[0] if isinstance(content[0], dict) and content[0].get("type") == "Summary" else {}
    samples = [item for item in content if isinstance(item, dict) and item.get("type") != "Summary"]
    return summary, samples


def build_report(summary: Dict[str, Any], samples: List[Dict[str, Any]], top_k: int, failure_top_k: int) -> Dict[str, Any]:
    report = {
        "input_summary": summary,
        "sample_count": len(samples),
        "timing": summarize_times(samples),
        "answer_quality": analyze_answer_quality(samples),
        "reasoning": analyze_reasoning(samples),
        "retrieval_diagnostics": analyze_retrieval_failures(samples, top_failure_count=failure_top_k),
        "slowest_samples": get_slowest_samples(samples, top_k=top_k),
        "notes": [
            "The original eval summary stores path generation time in the third slot of time[].",
            "In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.",
            "avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.",
            "Predictions are normalized to strip SubQuery/SubAnswer/Final Answer wrappers before answer-quality scoring.",
            "Retrieval diagnostics now separate first-hop failure, rewrite salvage, relation drift, category-specific failure rates, and result-format anomalies.",
        ],
    }

    if any(sample.get("corag_recall") for sample in samples):
        report["corag_recall"] = analyze_recall(samples, "corag_recall")
    if any(sample.get("naive_recall") for sample in samples):
        report["naive_recall"] = analyze_recall(samples, "naive_recall")
        report["recall_comparison"] = compare_recall(samples)

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("=== Eval Analysis ===")
    print(f"Samples: {report['sample_count']}")

    answer_quality = report["answer_quality"]
    print("\n[Answer Quality]")
    print(f"EM: {answer_quality['exact_match']:.4f}")
    print(f"Token F1: {answer_quality['token_f1']:.4f}")
    print(f"Contains Match: {answer_quality['contains_match']:.4f}")
    print(f"Yes/No Accuracy: {answer_quality['yes_no_accuracy']:.4f} (n={answer_quality['yes_no_total']})")
    print(f"No Relevant Information Rate: {answer_quality['no_relevant_info_rate']:.4f}")

    print("\n[Timing]")
    for key, stats in report["timing"].items():
        print(
            f"{key}: mean={stats['mean']:.2f}s, median={stats['median']:.2f}s, "
            f"p90={stats['p90']:.2f}s, p95={stats['p95']:.2f}s, max={stats['max']:.2f}s"
        )

    print("\n[Reasoning]")
    reasoning = report["reasoning"]
    print(
        f"avg_steps={reasoning['avg_steps']:.2f}, median_steps={reasoning['median_steps']:.2f}, "
        f"min_steps={reasoning['min_steps']}, max_steps={reasoning['max_steps']}"
    )

    diagnostics = report["retrieval_diagnostics"]
    print("\n[Retrieval Diagnostics]")
    print(f"zero_corag_recall_rate={diagnostics['zero_corag_recall_rate']:.4f}")
    print(f"no_info_rate={diagnostics['no_info_rate']:.4f}")
    print(f"yes_no_error_rate={diagnostics['yes_no_error_rate']:.4f}")
    print(f"first_step_failure_rate={diagnostics['first_step_failure_rate']:.4f}")
    print(f"rewrite_after_failure_rate={diagnostics['rewrite_after_failure_rate']:.4f}")
    print(f"rewrite_salvage_rate={diagnostics['rewrite_salvage_rate']:.4f}")
    print(f"relation_drift_count={diagnostics['relation_drift_count']}")
    print(f"comparison_one_side_missing_rate={diagnostics['comparison_one_side_missing_rate']:.4f}")
    print(f"complex_kinship_failure_rate={diagnostics['complex_kinship_failure_rate']:.4f}")
    print(f"role_bridge_failure_rate={diagnostics['role_bridge_failure_rate']:.4f}")
    print(f"corag_format_issue_total={diagnostics['corag_format_issue_total']}")
    print(f"naive_format_issue_total={diagnostics['naive_format_issue_total']}")

    category_stats = diagnostics.get("category_stats", {})
    if category_stats:
        print("\n[Category Breakdown]")
        for category, stats in category_stats.items():
            print(
                f"{category}: count={int(stats['count'])}, "
                f"zero_recall_rate={stats['zero_recall_rate']:.4f}, "
                f"avg_corag_recall={stats['avg_corag_recall']:.4f}"
            )

    if "corag_recall" in report:
        recall = report["corag_recall"]
        print("\n[CoRAG Recall]")
        print(f"micro={recall['micro_recall']:.4f}, macro={recall['macro_recall']:.4f}, hits={recall['total_hits']}, gold={recall['total_gold']}")

    if "naive_recall" in report:
        recall = report["naive_recall"]
        comparison = report.get("recall_comparison", {})
        print("\n[Naive Recall]")
        print(f"micro={recall['micro_recall']:.4f}, macro={recall['macro_recall']:.4f}, hits={recall['total_hits']}, gold={recall['total_gold']}")
        print(
            f"Recall comparison: corag_better={comparison.get('corag_better', 0)}, "
            f"naive_better={comparison.get('naive_better', 0)}, tie={comparison.get('tie', 0)}"
        )

    print("\n[Top Failure Examples]")
    for index, sample in enumerate(diagnostics.get("failure_examples", []), start=1):
        print(
            f"{index}. recall={sample['corag_recall']:.4f}, "
            f"question={sample['question']}, "
            f"first_subquery={sample['first_subquery']}, "
            f"first_subanswer={sample['first_subanswer']}"
        )

    print("\n[Slowest Samples]")
    for index, sample in enumerate(report["slowest_samples"], start=1):
        print(
            f"{index}. total={sample['total_time']:.2f}s, path={sample['path_generation_time']:.2f}s, "
            f"final={sample['final_generation_time']:.2f}s, steps={sample['reasoning_steps']}, question={sample['question']}"
        )

    print("\n[Notes]")
    for note in report["notes"]:
        print(f"- {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze custom_batch_eval output JSON.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="",
        help="Path to eval output JSON file. If omitted, DEFAULT_EVAL_FILE in this script will be used.",
    )
    parser.add_argument("--save_report", type=str, default="", help="Optional path to save analysis report JSON")
    parser.add_argument("--top_k", type=int, default=10, help="Number of slowest samples to include")
    parser.add_argument("--failure_top_k", type=int, default=8, help="Number of failure examples to include")
    args = parser.parse_args()

    input_path = Path(args.input_file) if args.input_file else DEFAULT_EVAL_FILE
    resolved_input_path = input_path.resolve()
    print(f"Input eval file: {resolved_input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Eval output JSON not found: {resolved_input_path}")

    summary, samples = load_eval_file(input_path)
    report = build_report(summary, samples, top_k=args.top_k, failure_top_k=args.failure_top_k)
    print_report(report)

    if args.save_report:
        output_path = Path(args.save_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
