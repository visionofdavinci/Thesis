"""
Ablation Analysis Runner

Processes each ablation end-to-end, producing for each one:
    <ablation>_tables.txt          plain-text tables, one per scenario
    <ablation>_tables.tex          LaTeX tables ready for inclusion
    <ablation>__<scenario>.csv     one CSV per scenario (raw means + sig flags)

Each ablation is processed independently. No single output file mixes data
across ablations.
"""

from __future__ import annotations

import argparse
import os
import sys

#ensure local imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_tables import process_ablation, METRICS_BY_ABLATION


ABLATIONS = [
    {
        "name":    "ablation_1_subharmonic",
        "label":   "abl1",
        "caption": "Ablation 1 (subharmonic; RQ1 smoothness, RQ2 convergence)",
    },
    {
        "name":    "ablation_2_superharmonic",
        "label":   "abl2",
        "caption": "Ablation 2 (superharmonic; RQ3 temporal reactivity)",
    },
    {
        "name":    "ablation_3_ppo_escape",
        "label":   "abl3",
        "caption": "Ablation 3 (PPO escape; RQ4 freeze-state recovery)",
    },
    {
        "name":    "ablation_4_full_integration",
        "label":   "abl4",
        "caption": "Ablation 4 (full integration; all RQs)",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Process ablation results into readable tables")
    parser.add_argument("--input-dir",  default="evaluation/results",
                        help="Directory with ablation_*_results.json and ablation_*_tests.json")
    parser.add_argument("--output-dir", default="evaluation/output",
                        help="Where to write processed tables and interpretations")
    parser.add_argument("--alpha",      type=float, default=0.05,
                        help="Significance level for FDR-corrected p-values")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing ablations from {args.input_dir}")
    print(f"Writing outputs to {args.output_dir}")
    print(f"Significance threshold: p_adj < {args.alpha}")
    print()

    for abl in ABLATIONS:
        name = abl["name"]
        results_path = os.path.join(args.input_dir, f"{name}_results.json")
        tests_path   = os.path.join(args.input_dir, f"{name}_tests.json")
        if not os.path.exists(results_path):
            print(f"  [skip] {name}: results file not found at {results_path}")
            continue

        metric_specs = METRICS_BY_ABLATION[name]
        product = process_ablation(
            results_path=results_path,
            tests_path=tests_path if os.path.exists(tests_path) else None,
            output_dir=args.output_dir,
            metric_specs=metric_specs,
            ablation_label=abl["label"],
            ablation_caption_prefix=abl["caption"],
            alpha=args.alpha,
        )


        print(f"  [ok]   {name}")
        print(f"           tables txt: {product.text_path}")
        print(f"           tables tex: {product.latex_path}")
        for p in product.csv_paths:
            print(f"           csv       : {p}")
        print(f"           interpret : {product.interpretation_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
