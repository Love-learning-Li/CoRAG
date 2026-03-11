import argparse
import json
import math
import re
import string
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple


DEFAULT_EVAL_FILE = Path(r"G:/大学资料/3_研0/InternProject/CoRAG/Agentic_CoRAG-corag/eval/rejection_sampled_data_small_eval_out.json")


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
    return float(norm_gt in norm_pred or norm_pred in norm_gt)


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


def summarize_times(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    path_times = [sample["time"][2] for sample in samples]
    final_times = [sample["time"][3] for sample in samples]
    total_times = [sample["time"][2] + sample["time"][3] for sample in samples]

    return {
        "path_generation": build_distribution(path_times),
        "final_generation": build_distribution(final_times),
        "end_to_end": build_distribution(total_times),
    }


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


def analyze_answer_quality(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    em_scores: List[float] = []
    f1_scores: List[float] = []
    contains_scores: List[float] = []
    yesno_total = 0
    yesno_correct = 0
    no_info_count = 0

    for sample in samples:
        prediction = sample.get("answer", "")
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

    return {
        "corag_better": corag_better,
        "naive_better": naive_better,
        "tie": tie,
    }


def analyze_reasoning(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    step_counts = [len(sample.get("reasoning_steps", [])) for sample in samples]
    return {
        "avg_steps": mean(step_counts) if step_counts else 0.0,
        "median_steps": median(step_counts) if step_counts else 0.0,
        "max_steps": max(step_counts) if step_counts else 0,
        "min_steps": min(step_counts) if step_counts else 0,
    }


def analyze_retrieval_failures(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(samples)
    zero_corag = [sample for sample in samples if sample.get("corag_recall", {}).get("recall", 0.0) == 0.0]
    no_info = [sample for sample in samples if "no relevant information found" in sample.get("answer", "").lower()]
    yes_no_errors = [
        sample for sample in samples
        if normalize_squad(sample.get("ground_truth", "")) in {"yes", "no"}
        and normalize_squad(sample.get("answer", "")) != normalize_squad(sample.get("ground_truth", ""))
    ]

    relation_drift = 0
    for sample in samples:
        question = sample.get("question", "").lower()
        steps = sample.get("reasoning_steps", [])
        if not steps:
            continue
        step_queries = " ".join(step.get("subquery", "").lower() for step in steps)
        if "father-in-law" in question and "spouse" not in step_queries:
            relation_drift += 1
        elif "maternal grandmother" in question and "mother" not in step_queries:
            relation_drift += 1
        elif "paternal grandmother" in question and "father" not in step_queries:
            relation_drift += 1
        elif "stepmother" in question and "father" not in step_queries:
            relation_drift += 1

    return {
        "zero_corag_recall_rate": len(zero_corag) / total if total else 0.0,
        "no_info_rate": len(no_info) / total if total else 0.0,
        "yes_no_error_rate": len(yes_no_errors) / len([s for s in samples if normalize_squad(s.get("ground_truth", "")) in {"yes", "no"}]) if any(normalize_squad(s.get("ground_truth", "")) in {"yes", "no"} for s in samples) else 0.0,
        "relation_drift_count": relation_drift,
        "examples": [
            {
                "question": sample.get("question", ""),
                "prediction": sample.get("answer", ""),
                "ground_truth": sample.get("ground_truth", ""),
                "corag_recall": sample.get("corag_recall", {}).get("recall", 0.0),
            }
            for sample in zero_corag[:5]
        ],
    }


def get_slowest_samples(samples: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        samples,
        key=lambda sample: sample["time"][2] + sample["time"][3],
        reverse=True,
    )
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


def build_report(summary: Dict[str, Any], samples: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    report = {
        "input_summary": summary,
        "sample_count": len(samples),
        "timing": summarize_times(samples),
        "answer_quality": analyze_answer_quality(samples),
        "reasoning": analyze_reasoning(samples),
        "retrieval_diagnostics": analyze_retrieval_failures(samples),
        "slowest_samples": get_slowest_samples(samples, top_k=top_k),
        "notes": [
            "The original eval summary stores path generation time in the third slot of time[].",
            "In custom_batch_eval.py, avg_reranker_time is currently used to summarize path generation time rather than an actual reranker stage.",
            "avg_llm_call_time in the original summary corresponds to final answer generation time, not total LLM time across all sub-steps.",
            "Low recall often comes from subquery relation drift or overly broad entity resolution before retrieval, not only from the retriever itself.",
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
    print(f"relation_drift_count={diagnostics['relation_drift_count']}")

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
    args = parser.parse_args()

    input_path = Path(args.input_file) if args.input_file else DEFAULT_EVAL_FILE
    resolved_input_path = input_path.resolve()
    print(f"Input eval file: {resolved_input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Eval output JSON not found: {resolved_input_path}")

    summary, samples = load_eval_file(input_path)
    report = build_report(summary, samples, top_k=args.top_k)
    print_report(report)

    if args.save_report:
        output_path = Path(args.save_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()