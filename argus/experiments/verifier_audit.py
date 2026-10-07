"""Stage 5.6 & 5.7: Verifier Audit Benchmark & Prompt Calibration Comparison.

Evaluates the Verifier agent across 16 challenging cases:
  - Exact support
  - Semantic paraphrase
  - Multi-evidence joint proof
  - Direct & quantitative contradictions
  - Scope / model organism overreach
  - Causal overreach
  - Speculative wording
  - Partial support & missing evidence

Supports comparing Prompt V1 (Strict Literal) vs Prompt V2 (Calibrated Entailment).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from app.agents.verifier import Verifier
from app.models.research import Claim, Evidence, VerificationStatus


# ---------------------------------------------------------------------------
# Benchmark Case Definition (16 Fixed Cases — Untouched)
# ---------------------------------------------------------------------------

@dataclass
class AuditCase:
    id: int
    category: str
    description: str
    claim_text: str
    quotes: List[str]
    expected_status: VerificationStatus


BENCHMARK_CASES: List[AuditCase] = [
    # --- Category 1: Direct Support (VERIFIED) ---
    AuditCase(
        id=1,
        category="Direct Support",
        description="Verbatim exact match of core assertion",
        claim_text="Cas9 requires an adjacent PAM sequence for target binding.",
        quotes=[
            "Cas9 requires recognition of a PAM sequence adjacent to the target site for target binding."
        ],
        expected_status=VerificationStatus.VERIFIED,
    ),
    AuditCase(
        id=2,
        category="Direct Support",
        description="Clean semantic paraphrase with identical factual scope",
        claim_text="The seed region of the guide RNA is critical for Cas9 cleavage fidelity.",
        quotes=[
            "Mismatches within the guide RNA seed region drastically impair the cleavage fidelity of the Cas9 enzyme."
        ],
        expected_status=VerificationStatus.VERIFIED,
    ),
    AuditCase(
        id=3,
        category="Direct Support",
        description="Multi-evidence quote where second quote provides the proof",
        claim_text="SpCas9 specifically targets canonical NGG motifs in double-stranded DNA.",
        quotes=[
            "Engineered variants of Cas nucleases can expand targetable ranges across genomic loci.",
            "In wild-type Streptococcus pyogenes Cas9 (SpCas9), canonical target recognition specifically requires an NGG motif in double-stranded DNA.",
        ],
        expected_status=VerificationStatus.VERIFIED,
    ),
    AuditCase(
        id=4,
        category="Direct Support",
        description="Collective multi-evidence synthesis",
        claim_text="High-fidelity Cas9 variants reduce off-target cleavage while preserving on-target efficacy.",
        quotes=[
            "Engineered high-fidelity Cas9 variants substantially lower off-target cleavage genome-wide.",
            "These high-fidelity nucleases maintain comparable on-target cleavage efficacy relative to wild-type enzymes.",
        ],
        expected_status=VerificationStatus.VERIFIED,
    ),

    # --- Category 2: Direct Contradiction (REJECTED) ---
    AuditCase(
        id=5,
        category="Contradiction",
        description="Direct functional negation of effect",
        claim_text="Calcium ions significantly enhance Cas9 cleavage activity.",
        quotes=[
            "Cas9 cleavage activity is completely inhibited and blocked in the presence of calcium ions."
        ],
        expected_status=VerificationStatus.REJECTED,
    ),
    AuditCase(
        id=6,
        category="Contradiction",
        description="Direct numerical / quantitative contradiction",
        claim_text="The engineered variant retained over 90% of wild-type kinetic efficiency.",
        quotes=[
            "Kinetic measurements revealed that the engineered mutant retained less than 15% of wild-type catalytic efficiency."
        ],
        expected_status=VerificationStatus.REJECTED,
    ),
    AuditCase(
        id=7,
        category="Contradiction",
        description="Universal negative claim contradicted by factual observation",
        claim_text="Cas9 never cleaves DNA sequences containing seed-region mismatches.",
        quotes=[
            "Genome-wide profiling confirmed that Cas9 readily cleaves certain non-target loci despite harboring single seed-region mismatches."
        ],
        expected_status=VerificationStatus.REJECTED,
    ),
    AuditCase(
        id=8,
        category="Contradiction",
        description="Claim asserts permanence when evidence states reversibility",
        claim_text="The chemical modification permanently inactivates the Cas9 ribonucleoprotein complex.",
        quotes=[
            "Upon exposure to visible light, the chemical modification is cleaved, fully restoring ribonucleoprotein activity within minutes."
        ],
        expected_status=VerificationStatus.REJECTED,
    ),

    # --- Category 3: Scope Overreach & Extrapolation (UNCERTAIN) ---
    AuditCase(
        id=9,
        category="Scope Overreach",
        description="Model organism overreach (mice findings asserted as human clinical fact)",
        claim_text="The gene therapy successfully reverses liver fibrosis in human clinical patients.",
        quotes=[
            "In a murine model of chronic liver injury, the therapeutic vector successfully reversed hepatic fibrosis in mice."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=10,
        category="Causal Overreach",
        description="Correlation presented as direct causation",
        claim_text="Elevated Cas9 expression directly causes increased chromosomal translocations.",
        quotes=[
            "High levels of Cas9 expression were positively correlated with higher observed frequencies of chromosomal translocations."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=11,
        category="Speculative Wording",
        description="Hypothetical suggestion claimed as verified truth",
        claim_text="The loop region determines the binding affinity for modified tracrRNA.",
        quotes=[
            "We speculate that this flexible loop region may participate in governing binding affinity for modified tracrRNA."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=12,
        category="Quantitative Overreach",
        description="Specific metric asserted without numeric support in evidence",
        claim_text="The treatment increased cell viability by exactly 45 percent.",
        quotes=[
            "Treated cohorts exhibited a statistically significant and noticeable increase in overall cell viability."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),

    # --- Category 4: Partial / Misleading / Missing Evidence (UNCERTAIN) ---
    AuditCase(
        id=13,
        category="Partial Support",
        description="Compound claim where quote only supports the first half",
        claim_text="Cas9 requires ATP hydrolysis and operates optimally at physiological pH.",
        quotes=[
            "Biochemical assays demonstrated that Cas9 functions optimally at physiological pH 7.4."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=14,
        category="Irrelevant Evidence",
        description="Evidence discussion is on a completely different topic",
        claim_text="Cas9 undergoes conformational change upon guide RNA loading.",
        quotes=[
            "Photosynthetic efficiency in Arabidopsis thaliana is regulated by non-photochemical quenching under excess irradiance."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=15,
        category="Outside Knowledge Trap",
        description="True in real-world biology, but absent from provided text",
        claim_text="The human genome contains approximately 3 billion base pairs.",
        quotes=[
            "Streptococcus pyogenes is a Gram-positive pathogen that harbors a Type II CRISPR-Cas adaptive immune system."
        ],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
    AuditCase(
        id=16,
        category="Missing Evidence",
        description="Zero evidence quotes provided",
        claim_text="CRISPR interference selectively silences gene transcription without cutting DNA.",
        quotes=[],
        expected_status=VerificationStatus.UNCERTAIN,
    ),
]


# ---------------------------------------------------------------------------
# Benchmark Execution & Comparison Logic
# ---------------------------------------------------------------------------

def run_audit(prompt_version: str = "v2", model: str | None = None) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = model or os.getenv("DEFAULT_LLM_MODEL", "llama3.2:3b")

    print("=" * 76)
    print(f"  ARGUS VERIFIER AUDIT BENCHMARK — PROMPT {prompt_version.upper()}")
    print(f"  Provider       : {provider}")
    print(f"  Model          : {model_name}")
    print(f"  Prompt Version : {prompt_version.upper()}")
    print(f"  Test Cases     : {len(BENCHMARK_CASES)}")
    print("=" * 76)

    verifier = Verifier(model=model_name, prompt_version=prompt_version)

    statuses = [
        VerificationStatus.VERIFIED,
        VerificationStatus.REJECTED,
        VerificationStatus.UNCERTAIN,
    ]
    matrix = {exp: {pred: 0 for pred in statuses} for exp in statuses}

    results_log = []
    correct_count = 0
    start_time = time.time()

    for case in BENCHMARK_CASES:
        evidence_items = [
            Evidence(source_id="audit-src", quote=q)
            for q in case.quotes
        ]
        claim = Claim(claim_text=case.claim_text)

        t0 = time.time()
        try:
            v_res = verifier.verify_claim(claim, evidence_items)
        except (TimeoutError, Exception) as exc:
            print(f"      [RETRY] Case #{case.id} encountered {exc}. Retrying once in 3s...")
            time.sleep(3.0)
            v_res = verifier.verify_claim(claim, evidence_items)
        elapsed = time.time() - t0
        time.sleep(0.5)

        is_correct = (v_res.status == case.expected_status)
        if is_correct:
            correct_count += 1

        matrix[case.expected_status][v_res.status] += 1

        results_log.append({
            "id": case.id,
            "category": case.category,
            "claim": case.claim_text,
            "expected": case.expected_status.value,
            "predicted": v_res.status.value,
            "confidence": round(v_res.confidence, 2),
            "is_correct": is_correct,
            "reasoning": v_res.reasoning,
            "elapsed": round(elapsed, 2),
        })

        icon = "[PASS]" if is_correct else "[FAIL]"
        print(f"\n{icon} Case #{case.id:02d} [{case.category}] ({elapsed:.2f}s)")
        print(f"      Claim    : \"{case.claim_text}\"")
        print(f"      Expected : {case.expected_status.value.upper()}")
        print(f"      Predicted: {v_res.status.value.upper()} (conf={v_res.confidence:.2f})")
        print(f"      Reasoning: {v_res.reasoning}")

    total_time = time.time() - start_time
    accuracy = (correct_count / len(BENCHMARK_CASES)) * 100

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

    # False VERIFIED count (critical metric)
    false_verified = sum(
        matrix[exp][VerificationStatus.VERIFIED]
        for exp in (VerificationStatus.REJECTED, VerificationStatus.UNCERTAIN)
    )

    report_payload = {
        "prompt_version": prompt_version,
        "model": model_name,
        "total_cases": len(BENCHMARK_CASES),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 2),
        "false_verified_count": false_verified,
        "confusion_matrix": {
            exp.value: {pred.value: matrix[exp][pred] for pred in statuses}
            for exp in statuses
        },
        "metrics": class_metrics,
        "results": results_log,
    }

    # Save to JSON
    out_file = Path(__file__).resolve().parent / f"verifier_results_{prompt_version}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[OK] Saved audit results to: {out_file.name}")

    # Print summary
    print("\n" + "=" * 76)
    print(f"  CONFUSION MATRIX ({prompt_version.upper()})")
    print("=" * 76)
    header = f"{'Expected \\ Pred':<18} | {'VERIFIED':<10} | {'REJECTED':<10} | {'UNCERTAIN':<10} | {'Total':<6}"
    print(header)
    print("-" * len(header))
    for exp in statuses:
        row_counts = [matrix[exp][pred] for pred in statuses]
        print(f"{exp.value.upper():<18} | {row_counts[0]:<10} | {row_counts[1]:<10} | {row_counts[2]:<10} | {sum(row_counts):<6}")

    print("\n" + "=" * 76)
    print(f"  CLASS-LEVEL METRICS ({prompt_version.upper()})")
    print("=" * 76)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 58)
    for st in statuses:
        m = class_metrics[st.value]
        print(f"{st.value.upper():<12} | {m['precision']:>9.2%} | {m['recall']:>9.2%} | {m['f1']:>9.2%} | {m['support']:>8}")

    print("-" * 58)
    print(f"Overall Accuracy   : {correct_count}/{len(BENCHMARK_CASES)} ({accuracy:.1f}%)")
    print(f"False VERIFIED (FP): {false_verified}")
    print(f"Total Audit Time   : {total_time:.1f}s")
    print("=" * 76)

    # Check for V1 baseline to print comparative scorecard
    v1_file = Path(__file__).resolve().parent / "verifier_results_v1.json"
    if prompt_version == "v2" and v1_file.exists():
        try:
            with open(v1_file, "r", encoding="utf-8") as vf:
                v1_data = json.load(vf)
            print_comparison_scorecard(v1_data, report_payload)
        except Exception as e:
            logger.warning("Could not render comparative scorecard: %s", e)

    return report_payload


def print_comparison_scorecard(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    print("\n" + "=" * 76)
    print("  STAGE 5.7: PROMPT CALIBRATION COMPARATIVE SCORECARD (V1 vs V2)")
    print("=" * 76)
    hdr = f"{'Metric':<25} | {'V1 (Strict)':<16} | {'V2 (Calibrated)':<16} | {'Delta':<10}"
    print(hdr)
    print("-" * len(hdr))

    def fmt_pct(val):
        return f"{val:.1%}" if isinstance(val, (int, float)) else str(val)

    # Accuracy
    v1_acc = v1["accuracy"] / 100.0
    v2_acc = v2["accuracy"] / 100.0
    delta_acc = v2_acc - v1_acc
    print(f"{'Accuracy':<25} | {fmt_pct(v1_acc):<16} | {fmt_pct(v2_acc):<16} | {f'{delta_acc:+.1%}':<10}")

    # VERIFIED
    v1_vp = v1["metrics"]["verified"]["precision"]
    v2_vp = v2["metrics"]["verified"]["precision"]
    print(f"{'VERIFIED Precision':<25} | {fmt_pct(v1_vp):<16} | {fmt_pct(v2_vp):<16} | {f'{(v2_vp - v1_vp):+.1%}':<10}")

    v1_vr = v1["metrics"]["verified"]["recall"]
    v2_vr = v2["metrics"]["verified"]["recall"]
    print(f"{'VERIFIED Recall':<25} | {fmt_pct(v1_vr):<16} | {fmt_pct(v2_vr):<16} | {f'{(v2_vr - v1_vr):+.1%}':<10}")

    # REJECTED
    v1_rp = v1["metrics"]["rejected"]["precision"]
    v2_rp = v2["metrics"]["rejected"]["precision"]
    print(f"{'REJECTED Precision':<25} | {fmt_pct(v1_rp):<16} | {fmt_pct(v2_rp):<16} | {f'{(v2_rp - v1_rp):+.1%}':<10}")

    v1_rr = v1["metrics"]["rejected"]["recall"]
    v2_rr = v2["metrics"]["rejected"]["recall"]
    print(f"{'REJECTED Recall':<25} | {fmt_pct(v1_rr):<16} | {fmt_pct(v2_rr):<16} | {f'{(v2_rr - v1_rr):+.1%}':<10}")

    # UNCERTAIN Recall
    v1_ur = v1["metrics"]["uncertain"]["recall"]
    v2_ur = v2["metrics"]["uncertain"]["recall"]
    print(f"{'UNCERTAIN Recall':<25} | {fmt_pct(v1_ur):<16} | {fmt_pct(v2_ur):<16} | {f'{(v2_ur - v1_ur):+.1%}':<10}")

    # False VERIFIED Count
    v1_fv = v1.get("false_verified_count", 0)
    v2_fv = v2.get("false_verified_count", 0)
    print(f"{'False VERIFIED (FP)':<25} | {str(v1_fv):<16} | {str(v2_fv):<16} | {f'{(v2_fv - v1_fv):+d}':<10}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARGUS Verifier Audit Benchmark")
    parser.add_argument(
        "--prompt-version",
        default="v2",
        choices=["v1", "v2"],
        help="Prompt version to evaluate (default: v2)",
    )
    args = parser.parse_args()
    run_audit(prompt_version=args.prompt_version)
