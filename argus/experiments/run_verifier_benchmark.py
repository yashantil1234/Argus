"""Benchmark runner for the 120-case Verifier Benchmark (Dev & Test splits).

Features:
  - Supports --split dev (80 cases) and --split test (40 cases)
  - Evaluates Prompt V1 (Strict) vs Prompt V2 (Calibrated) vs future V3
  - Checkpoints every case to disk (resumable)
  - Computes:
      * Overall Accuracy
      * Precision, Recall, F1 per class (VERIFIED, REJECTED, UNCERTAIN)
      * False-VERIFIED rate (Primary safety metric)
      * Breakdown by all 11 scientific/linguistic phenomena
      * Confidence calibration (Avg confidence on correct vs incorrect decisions)
      * Comparative scorecard (V1 vs V2) when available

Usage:
    python experiments/run_verifier_benchmark.py --split dev --prompt-version v1
    python experiments/run_verifier_benchmark.py --split dev --prompt-version v2
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
    resume: bool = True,
) -> dict[str, Any]:
    dataset = load_dataset(split)
    if limit:
        dataset = dataset[:limit]

    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = model or os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")
    safe_model = model_name.replace(":", "_").replace("/", "_")

    out_file = Path(__file__).resolve().parent / f"benchmark_{split}_{safe_model}_{prompt_version}.json"
    checkpoint_file = Path(__file__).resolve().parent / f".checkpoint_{split}_{safe_model}_{prompt_version}.json"

    # Resume from checkpoint if present
    results_log: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results_log = saved.get("results", [])
                completed_ids = {r["id"] for r in results_log}
                print(f"[RESUME] Loaded {len(completed_ids)} cases from checkpoint {checkpoint_file.name}")
        except Exception:
            results_log = []
            completed_ids = set()

    print("=" * 82)
    print(f"  ARGUS VERIFIER BENCHMARK: SPLIT '{split.upper()}' ({len(dataset)} total cases)")
    print(f"  Provider       : {provider}")
    print(f"  Model          : {model_name}")
    print(f"  Prompt Version : {prompt_version.upper()}")
    print(f"  Remaining Cases: {len(dataset) - len(completed_ids)}")
    print("=" * 82)

    verifier = Verifier(model=model_name, prompt_version=prompt_version)

    statuses = [
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
        VerificationStatus.UNCERTAIN,
    ]

    start_time = time.time()

    for i, item in enumerate(dataset):
        case_id = item["id"]
        if case_id in completed_ids:
            continue

        expected_status = VerificationStatus(item["label"])
        evidence_items = [
            Evidence(source_id=f"src_{case_id}", quote=q)
            for q in item["quotes"]
        ]
        claim = Claim(claim_text=item["claim_text"])

        t0 = time.time()
        try:
            v_res = verifier.verify_claim(claim, evidence_items)
        except Exception as exc:
            print(f"  [RETRY] Case #{case_id} error: {exc}. Retrying in 2s...")
            time.sleep(2.0)
            v_res = verifier.verify_claim(claim, evidence_items)

        elapsed = time.time() - t0
        time.sleep(0.3)  # brief thermal pacing pause

        is_correct = (v_res.status == expected_status)
        phen = item.get("phenomenon", "general")

        result_entry = {
            "id": case_id,
            "phenomenon": phen,
            "domain": item.get("domain", "general"),
            "claim": item["claim_text"],
            "expected": expected_status.value,
            "predicted": v_res.status.value,
            "confidence": round(v_res.confidence, 2),
            "is_correct": is_correct,
            "reasoning": v_res.reasoning,
            "elapsed": round(elapsed, 2),
        }
        results_log.append(result_entry)
        completed_ids.add(case_id)

        # Save checkpoint periodically
        with open(checkpoint_file, "w", encoding="utf-8") as cf:
            json.dump({"results": results_log}, cf, indent=2)

        icon = "[PASS]" if is_correct else "[FAIL]"
        print(f"[{len(results_log):02d}/{len(dataset):02d}] {icon} {case_id} ({phen}) "
              f"Exp={expected_status.value.upper()} Pred={v_res.status.value.upper()} "
              f"({elapsed:.1f}s)")

    total_time = time.time() - start_time

    # -----------------------------------------------------------------------
    # Compute Comprehensive Metrics
    # -----------------------------------------------------------------------
    matrix = {exp: {pred: 0 for pred in statuses} for exp in statuses}
    phenomenon_stats: dict[str, dict[str, int]] = {}
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    correct_count = 0

    for r in results_log:
        exp_st = VerificationStatus(r["expected"])
        pred_st = VerificationStatus(r["predicted"])
        matrix[exp_st][pred_st] += 1

        if r["is_correct"]:
            correct_count += 1
            correct_confidences.append(r["confidence"])
        else:
            incorrect_confidences.append(r["confidence"])

        ph = r.get("phenomenon", "general")
        if ph not in phenomenon_stats:
            phenomenon_stats[ph] = {"correct": 0, "total": 0}
        phenomenon_stats[ph]["total"] += 1
        if r["is_correct"]:
            phenomenon_stats[ph]["correct"] += 1

    accuracy = (correct_count / len(results_log)) * 100 if results_log else 0.0

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

    avg_conf_correct = (
        round(sum(correct_confidences) / len(correct_confidences), 4)
        if correct_confidences
        else 0.0
    )
    avg_conf_incorrect = (
        round(sum(incorrect_confidences) / len(incorrect_confidences), 4)
        if incorrect_confidences
        else 0.0
    )

    report_payload = {
        "split": split,
        "prompt_version": prompt_version,
        "model": model_name,
        "total_cases": len(results_log),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 2),
        "false_verified_count": false_verified,
        "calibration": {
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
            "confidence_gap": round(avg_conf_correct - avg_conf_incorrect, 4),
        },
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
            for k, v in sorted(phenomenon_stats.items())
        },
        "results": results_log,
    }

    # Save final report to JSON and remove checkpoint
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # -----------------------------------------------------------------------
    # Print Full Terminal Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 82)
    print(f"  BENCHMARK REPORT: {split.upper()} ({model_name} | {prompt_version.upper()})")
    print("=" * 82)
    hdr = f"{'Expected \\ Pred':<18} | {'VERIFIED':<10} | {'REJECTED':<10} | {'UNCERTAIN':<10} | {'Total':<6}"
    print(hdr)
    print("-" * len(hdr))
    for exp in statuses:
        row_counts = [matrix[exp][pred] for pred in statuses]
        print(f"{exp.value.upper():<18} | {row_counts[0]:<10} | {row_counts[1]:<10} | {row_counts[2]:<10} | {sum(row_counts):<6}")

    print("\n" + "-" * 82)
    print(f"{'Class':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 60)
    for st in statuses:
        m = class_metrics[st.value]
        print(f"{st.value.upper():<14} | {m['precision']:>9.2%} | {m['recall']:>9.2%} | {m['f1']:>9.2%} | {m['support']:>8}")

    print("\n" + "-" * 82)
    print(f"{'Phenomenon Category':<32} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 64)
    for ph, stats in report_payload["phenomenon_stats"].items():
        print(f"{ph:<32} | {stats['correct']:<8} | {stats['total']:<8} | {stats['acc']:>8.1f}%")

    print("\n" + "-" * 82)
    print(f"Overall Accuracy         : {correct_count}/{len(results_log)} ({accuracy:.1f}%)")
    print(f"False VERIFIED (Halluc): {false_verified} (safety risk cases)")
    print(f"Avg Confidence (Correct) : {avg_conf_correct:.2f}")
    print(f"Avg Confidence (Incorrect): {avg_conf_incorrect:.2f}")
    print(f"Confidence Gap           : {avg_conf_correct - avg_conf_incorrect:+.2f}")
    print(f"Total Benchmark Time     : {total_time:.1f}s")
    print(f"Saved Final Report       : {out_file.name}")
    print("=" * 82)

    # Check for V1 baseline if we ran V2
    if prompt_version == "v2":
        v1_file = Path(__file__).resolve().parent / f"benchmark_{split}_{safe_model}_v1.json"
        if v1_file.exists():
            try:
                with open(v1_file, "r", encoding="utf-8") as vf:
                    v1_data = json.load(vf)
                print_comparison_scorecard(v1_data, report_payload)
            except Exception as e:
                logger.warning("Could not render comparative scorecard: %s", e)

    return report_payload


def print_comparison_scorecard(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    print("\n" + "=" * 82)
    print("  COMPARATIVE SCORECARD: PROMPT V1 (STRICT) vs PROMPT V2 (CALIBRATED)")
    print("=" * 82)
    hdr = f"{'Metric':<28} | {'V1 (Strict)':<16} | {'V2 (Calibrated)':<16} | {'Delta':<10}"
    print(hdr)
    print("-" * len(hdr))

    def fmt_pct(val):
        return f"{val:.1%}" if isinstance(val, (int, float)) else str(val)

    # Accuracy
    v1_acc = v1["accuracy"] / 100.0
    v2_acc = v2["accuracy"] / 100.0
    print(f"{'Overall Accuracy':<28} | {fmt_pct(v1_acc):<16} | {fmt_pct(v2_acc):<16} | {f'{(v2_acc - v1_acc):+.1%}':<10}")

    # VERIFIED
    v1_vp = v1["metrics"]["verified"]["precision"]
    v2_vp = v2["metrics"]["verified"]["precision"]
    print(f"{'VERIFIED Precision':<28} | {fmt_pct(v1_vp):<16} | {fmt_pct(v2_vp):<16} | {f'{(v2_vp - v1_vp):+.1%}':<10}")

    v1_vr = v1["metrics"]["verified"]["recall"]
    v2_vr = v2["metrics"]["verified"]["recall"]
    print(f"{'VERIFIED Recall':<28} | {fmt_pct(v1_vr):<16} | {fmt_pct(v2_vr):<16} | {f'{(v2_vr - v1_vr):+.1%}':<10}")

    # REJECTED
    v1_rp = v1["metrics"]["rejected"]["precision"]
    v2_rp = v2["metrics"]["rejected"]["precision"]
    print(f"{'REJECTED Precision':<28} | {fmt_pct(v1_rp):<16} | {fmt_pct(v2_rp):<16} | {f'{(v2_rp - v1_rp):+.1%}':<10}")

    v1_rr = v1["metrics"]["rejected"]["recall"]
    v2_rr = v2["metrics"]["rejected"]["recall"]
    print(f"{'REJECTED Recall':<28} | {fmt_pct(v1_rr):<16} | {fmt_pct(v2_rr):<16} | {f'{(v2_rr - v1_rr):+.1%}':<10}")

    # UNCERTAIN
    v1_ur = v1["metrics"]["uncertain"]["recall"]
    v2_ur = v2["metrics"]["uncertain"]["recall"]
    print(f"{'UNCERTAIN Recall':<28} | {fmt_pct(v1_ur):<16} | {fmt_pct(v2_ur):<16} | {f'{(v2_ur - v1_ur):+.1%}':<10}")

    # False VERIFIED
    v1_fv = v1.get("false_verified_count", 0)
    v2_fv = v2.get("false_verified_count", 0)
    print(f"{'False VERIFIED (Halluc FP)':<28} | {str(v1_fv):<16} | {str(v2_fv):<16} | {f'{(v2_fv - v1_fv):+d}':<10}")

    # Confidence gap
    v1_gap = v1.get("calibration", {}).get("confidence_gap", 0.0)
    v2_gap = v2.get("calibration", {}).get("confidence_gap", 0.0)
    print(f"{'Confidence Gap (Corr-Incorr)':<28} | {f'{v1_gap:+.2f}':<16} | {f'{v2_gap:+.2f}':<16} | {f'{(v2_gap - v1_gap):+.2f}':<10}")
    print("=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 120-Case Verifier Benchmark")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Dataset split (default: dev)")
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"], help="Prompt version (default: v2)")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming from checkpoint")
    args = parser.parse_args()

    run_benchmark(
        split=args.split,
        prompt_version=args.prompt_version,
        model=args.model,
        limit=args.limit,
        resume=not args.no_resume,
    )
