"""Benchmark runner for the 120-case Verifier Benchmark (Dev & Test splits).

Usage:
    python experiments/run_verifier_benchmark.py --split dev --prompt-version v1
    python experiments/run_verifier_benchmark.py --split dev --prompt-version v2
    python experiments/run_verifier_benchmark.py --split test --prompt-version v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from app.agents.verifier import Verifier
from app.models.research import Claim, Evidence, VerificationStatus


BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "verifier_benchmark"


def load_dataset(split: str) -> List[dict[str, Any]]:
    path = BENCHMARK_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark split not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    split: str = "dev",
    prompt_version: str = "v2",
    model: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(split)
    if limit:
        dataset = dataset[:limit]

    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = model or os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

    print("=" * 78)
    print(f"  ARGUS VERIFIER BENCHMARK: SPLIT '{split.upper()}' ({len(dataset)} cases)")
    print(f"  Provider       : {provider}")
    print(f"  Model          : {model_name}")
    print(f"  Prompt Version : {prompt_version.upper()}")
    print("=" * 78)

    verifier = Verifier(model=model_name, prompt_version=prompt_version)

    statuses = [
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
        VerificationStatus.UNCERTAIN,
    ]
    matrix = {exp: {pred: 0 for pred in statuses} for exp in statuses}
    phenomenon_stats: dict[str, dict[str, int]] = {}

    correct_count = 0
    results_log = []
    start_time = time.time()

    for i, item in enumerate(dataset):
        expected_status = VerificationStatus(item["label"])
        evidence_items = [
            Evidence(source_id=f"src_{item['id']}", quote=q)
            for q in item["quotes"]
        ]
        claim = Claim(claim_text=item["claim_text"])

        t0 = time.time()
        try:
            v_res = verifier.verify_claim(claim, evidence_items)
        except Exception as exc:
            print(f"  [RETRY] Case #{item['id']} error: {exc}. Retrying in 2s...")
            time.sleep(2.0)
            v_res = verifier.verify_claim(claim, evidence_items)

        elapsed = time.time() - t0
        time.sleep(0.3)  # pacing pause

        is_correct = (v_res.status == expected_status)
        if is_correct:
            correct_count += 1

        matrix[expected_status][v_res.status] += 1

        # Track per-phenomenon accuracy
        phen = item.get("phenomenon", "general")
        if phen not in phenomenon_stats:
            phenomenon_stats[phen] = {"correct": 0, "total": 0}
        phenomenon_stats[phen]["total"] += 1
        if is_correct:
            phenomenon_stats[phen]["correct"] += 1

        results_log.append({
            "id": item["id"],
            "phenomenon": phen,
            "domain": item.get("domain", "general"),
            "claim": item["claim_text"],
            "expected": expected_status.value,
            "predicted": v_res.status.value,
            "confidence": round(v_res.confidence, 2),
            "is_correct": is_correct,
            "reasoning": v_res.reasoning,
            "elapsed": round(elapsed, 2),
        })

        icon = "[PASS]" if is_correct else "[FAIL]"
        print(f"[{i + 1:02d}/{len(dataset):02d}] {icon} {item['id']} ({phen}) "
              f"Exp={expected_status.value.upper()} Pred={v_res.status.value.upper()} "
              f"({elapsed:.1f}s)")

    total_time = time.time() - start_time
    accuracy = (correct_count / len(dataset)) * 100

    # Calculate metrics
    class_metrics = {}
    for st in statuses:
        tp = matrix[st][st]
        fp = sum(matrix[other][st] for other in statuses if other != st)
        fn = sum(matrix[st][other] for other in statuses if other != st)
        support = sum(matrix[st][other] for other in statuses)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[st.value] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    false_verified = sum(
        matrix[exp][VerificationStatus.VERIFIED]
        for exp in (VerificationStatus.REJECTED, VerificationStatus.UNCERTAIN)
    )

    report_payload = {
        "split": split,
        "prompt_version": prompt_version,
        "model": model_name,
        "total_cases": len(dataset),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 2),
        "false_verified_count": false_verified,
        "confusion_matrix": {
            exp.value: {pred.value: matrix[exp][pred] for pred in statuses}
            for exp in statuses
        },
        "metrics": class_metrics,
        "phenomenon_stats": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "acc": round((v["correct"] / v["total"]) * 100, 1) if v["total"] > 0 else 0.0,
            }
            for k, v in phenomenon_stats.items()
        },
        "results": results_log,
    }

    # Save to JSON
    safe_model = model_name.replace(":", "_").replace("/", "_")
    out_file = Path(__file__).resolve().parent / f"benchmark_{split}_{safe_model}_{prompt_version}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Print Report
    print("\n" + "=" * 78)
    print(f"  BENCHMARK REPORT: {split.upper()} ({model_name} | {prompt_version.upper()})")
    print("=" * 78)
    hdr = f"{'Expected \\ Pred':<18} | {'VERIFIED':<10} | {'REJECTED':<10} | {'UNCERTAIN':<10} | {'Total':<6}"
    print(hdr)
    print("-" * len(hdr))
    for exp in statuses:
        row_counts = [matrix[exp][pred] for pred in statuses]
        print(f"{exp.value.upper():<18} | {row_counts[0]:<10} | {row_counts[1]:<10} | {row_counts[2]:<10} | {sum(row_counts):<6}")

    print("\n" + "-" * 78)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 58)
    for st in statuses:
        m = class_metrics[st.value]
        print(f"{st.value.upper():<12} | {m['precision']:>9.2%} | {m['recall']:>9.2%} | {m['f1']:>9.2%} | {m['support']:>8}")

    print("-" * 78)
    print(f"Overall Accuracy   : {correct_count}/{len(dataset)} ({accuracy:.1f}%)")
    print(f"False VERIFIED (FP): {false_verified}")
    print(f"Total Time         : {total_time:.1f}s (avg {total_time / len(dataset):.1f}s/case)")
    print(f"Saved Report       : {out_file.name}")
    print("=" * 78)

    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 120-Case Verifier Benchmark")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"], help="Prompt version")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()

    run_benchmark(
        split=args.split,
        prompt_version=args.prompt_version,
        model=args.model,
        limit=args.limit,
    )
