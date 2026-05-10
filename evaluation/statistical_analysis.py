"""
Statistical Analysis for Ablation Results
"""

import argparse
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


#statistical test implementations (no scipy dependency)

def wilcoxon_signed_rank(x, y):
    """
    Wilcoxon signed-rank test for paired samples.
    Tests H0: the distribution of x-y is symmetric around zero.

    Returns (W_statistic, p_value_approx, effect_size_r).
    Uses normal approximation for n >= 10.
    """
    d = np.array(x) - np.array(y)
    d = d[d != 0]  #remove zeros
    n = len(d)

    if n < 5:
        return np.nan, np.nan, np.nan

    ranks = _rank_data(np.abs(d))
    w_plus = np.sum(ranks[d > 0])
    w_minus = np.sum(ranks[d < 0])
    W = min(w_plus, w_minus)

    #normal approximation
    mean_W = n * (n + 1) / 4.0
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    if std_W < 1e-12:
        return W, 1.0, 0.0

    z = (W - mean_W) / std_W
    p = 2.0 * _normal_cdf(-abs(z))  #two-tailed

    #rank-biserial correlation as effect size
    r = 1.0 - (2.0 * W) / (n * (n + 1))

    return float(W), float(p), float(r)


def mcnemar_test(x, y):
    """
    McNemar's test for paired binary outcomes.
    x, y are arrays of 0/1 values (same length, paired by index).

    Returns (chi2, p_value, odds_ratio).
    """
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)

    #contingency table
    b = np.sum((x == 1) & (y == 0))  #x success, y failure
    c = np.sum((x == 0) & (y == 1))  #x failure, y success

    n_discord = b + c
    if n_discord == 0:
        return 0.0, 1.0, np.nan  #no disagreement

    #McNemar with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    #approximate p-value from chi-squared with df=1
    p = 1.0 - _chi2_cdf(chi2, df=1)

    #odds ratio
    odds = b / max(c, 1e-10)

    return float(chi2), float(p), float(odds)


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev, p * n / rank)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev = adj_p

    return adjusted


#helper functions

def _rank_data(data):
    """Assigns ranks to data, handling ties by averaging."""
    n = len(data)
    sorted_idx = np.argsort(data)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and data[sorted_idx[j + 1]] == data[sorted_idx[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _normal_cdf(x):
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    t = 1.0 / (1.0 + p * abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


def _chi2_cdf(x, df=1):
    """Approximate chi-squared CDF for df=1 using normal approximation."""
    if x <= 0:
        return 0.0
    z = np.sqrt(x)
    return 2.0 * _normal_cdf(z) - 1.0


# main analysis

def load_results(results_path: str) -> Dict:
    """Loads a saved ablation results JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def run_pairwise_tests(results: Dict, reference_config: str, metrics_to_test: List[str], binary_metrics: List[str] = None) -> List[Dict]:
    """
    Runs pairwise statistical tests: reference_config vs each other config.

    For continuous metrics: Wilcoxon signed-rank (paired by episode seed).
    For binary metrics: McNemar's test.

    Returns list of test result dicts.
    """
    if binary_metrics is None:
        binary_metrics = ["success", "collision"]

    #group results by scenario
    scenarios = set()
    configs = set()
    for key in results:
        config_name, scenario_name = key.split("__")
        scenarios.add(scenario_name)
        configs.add(config_name)

    test_results = []

    for scenario_name in sorted(scenarios):
        ref_key = f"{reference_config}__{scenario_name}"
        if ref_key not in results:
            continue
        ref_episodes = results[ref_key]["episode_metrics"]

        for config_name in sorted(configs):
            if config_name == reference_config:
                continue
            comp_key = f"{config_name}__{scenario_name}"
            if comp_key not in results:
                continue
            comp_episodes = results[comp_key]["episode_metrics"]

            n_ep = min(len(ref_episodes), len(comp_episodes))

            for metric in metrics_to_test:
                if metric in binary_metrics:
                    #McNemar's test
                    ref_vals = [ref_episodes[i].get(metric, 0) for i in range(n_ep)]
                    comp_vals = [comp_episodes[i].get(metric, 0) for i in range(n_ep)]
                    chi2, p_val, odds = mcnemar_test(ref_vals, comp_vals)
                    test_results.append({
                        "scenario": scenario_name,
                        "reference": reference_config,
                        "comparison": config_name,
                        "metric": metric,
                        "test": "McNemar",
                        "statistic": chi2,
                        "p_value": p_val,
                        "effect_size": odds,
                        "effect_label": "odds_ratio",
                        "ref_mean": np.mean(ref_vals),
                        "comp_mean": np.mean(comp_vals),
                    })
                else:
                    #Wilcoxon signed-rank test
                    ref_vals = [ref_episodes[i].get(metric, np.nan) for i in range(n_ep)]
                    comp_vals = [comp_episodes[i].get(metric, np.nan) for i in range(n_ep)]

                    #filter paired NaNs
                    valid_pairs = [(r, c) for r, c in zip(ref_vals, comp_vals) if not (np.isnan(r) or np.isnan(c) or r is None or c is None)]
                    if len(valid_pairs) < 10:
                        continue
                    rv, cv = zip(*valid_pairs)
                    W, p_val, r_eff = wilcoxon_signed_rank(list(rv), list(cv))
                    test_results.append({
                        "scenario": scenario_name,
                        "reference": reference_config,
                        "comparison": config_name,
                        "metric": metric,
                        "test": "Wilcoxon",
                        "statistic": W,
                        "p_value": p_val,
                        "effect_size": r_eff,
                        "effect_label": "rank_biserial_r",
                        "ref_mean": np.mean(rv),
                        "comp_mean": np.mean(cv),
                        "n_pairs": len(valid_pairs),
                    })

    #apply Benjamini-Hochberg FDR correction
    p_vals = [t["p_value"] for t in test_results if not np.isnan(t["p_value"])]
    if p_vals:
        adjusted = benjamini_hochberg(p_vals)
        j = 0
        for t in test_results:
            if not np.isnan(t["p_value"]):
                t["p_adjusted"] = adjusted[j]
                j += 1
            else:
                t["p_adjusted"] = np.nan

    return test_results


def format_results_table(test_results: List[Dict], alpha: float = 0.05) -> str:
    """Formats test results as a readable table."""
    lines = []
    lines.append(f"{'Scenario':<22} {'Ref vs Comp':<22} {'Metric':<24} "
                 f"{'Test':<10} {'p_adj':>8} {'Effect':>8} {'Sig':>4} "
                 f"{'Ref_mean':>10} {'Comp_mean':>10}")
    lines.append("-" * 120)

    for t in sorted(test_results, key=lambda x: (x["scenario"], x["metric"])):
        sig = "*" if t.get("p_adjusted", 1.0) < alpha else ""
        p_adj = t.get("p_adjusted", np.nan)
        p_str = f"{p_adj:.4f}" if not np.isnan(p_adj) else "N/A"
        eff = t.get("effect_size", np.nan)
        eff_str = f"{eff:.4f}" if not np.isnan(eff) else "N/A"
        ref_m = t.get("ref_mean", np.nan)
        comp_m = t.get("comp_mean", np.nan)
        ref_str = f"{ref_m:.4f}" if not np.isnan(ref_m) else "N/A"
        comp_str = f"{comp_m:.4f}" if not np.isnan(comp_m) else "N/A"

        lines.append(
            f"{t['scenario']:<22} {t['reference']+' vs '+t['comparison']:<22} "
            f"{t['metric']:<24} {t['test']:<10} {p_str:>8} {eff_str:>8} {sig:>4} "
            f"{ref_str:>10} {comp_str:>10}"
        )

    return "\n".join(lines)


def export_latex_table(test_results: List[Dict], alpha: float = 0.05) -> str:
    """Exports results as a LaTeX table fragment."""
    lines = []
    lines.append(r"\begin{tabular}{llllrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Comparison & Metric & Test & $p_\mathrm{adj}$ & "
                 r"Effect & $\overline{x}_\mathrm{ref}$ & $\overline{x}_\mathrm{comp}$ \\")
    lines.append(r"\midrule")

    for t in sorted(test_results, key=lambda x: (x["scenario"], x["metric"])):
        p_adj = t.get("p_adjusted", np.nan)
        sig = r"$^{*}$" if p_adj < alpha else ""
        p_str = f"{p_adj:.3f}" if not np.isnan(p_adj) else "---"
        eff = t.get("effect_size", np.nan)
        eff_str = f"{eff:.3f}" if not np.isnan(eff) else "---"
        ref_m = f"{t.get('ref_mean', np.nan):.3f}"
        comp_m = f"{t.get('comp_mean', np.nan):.3f}"

        lines.append(
            f"  {t['scenario']} & {t['comparison']} & {t['metric']} & "
            f"{t['test']} & {p_str}{sig} & {eff_str} & {ref_m} & {comp_m} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def analyse_ablation(results_path: str, output_dir: str, verbose: bool = True):
    """
    Full analysis pipeline for a single ablation study.
    """
    results = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)

    #infer ablation study from filename
    basename = os.path.basename(results_path).replace("_results.json", "")

    #determine reference config and metrics
    from evaluation.eval_config import ABLATION_STUDIES
    study = None
    for sid, s in ABLATION_STUDIES.items():
        if s.name == basename:
            study = s
            break

    if study is None:
        print(f"  Warning: could not match {basename} to a known ablation study.")
        print(f"  Using FULL as reference, testing all available metrics.")
        reference = "FULL"
        metrics = ["success", "collision", "log_dimensionless_jerk",
                    "velocity_variance", "time_to_goal", "spl",
                    "min_clearance", "recovery_rate", "detour_pct"]
    else:
        reference = "FULL"
        metrics = study.primary_metrics + ["success", "collision"]
        #deduplicate
        metrics = list(dict.fromkeys(metrics))

    binary_metrics = ["success", "collision"]

    test_results = run_pairwise_tests(results, reference, metrics, binary_metrics)

    #print table
    table = format_results_table(test_results)
    if verbose:
        print(f"\n{table}")

    #save plain text table
    with open(os.path.join(output_dir, f"{basename}_stats.txt"), "w", encoding="utf-8") as f:
        f.write(table)

    #save LaTeX table
    latex = export_latex_table(test_results)
    with open(os.path.join(output_dir, f"{basename}_stats.tex"), "w", encoding="utf-8") as f:
        f.write(latex)

    #save raw test results as JSON
    with open(os.path.join(output_dir, f"{basename}_tests.json"), "w") as f:
        json.dump(test_results, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)

    #summary of significant findings
    sig_results = [t for t in test_results if t.get("p_adjusted", 1.0) < 0.05]
    if verbose:
        print(f"\n  Significant findings (p_adj < 0.05): {len(sig_results)}/{len(test_results)}")
        for t in sig_results:
            direction = "FULL better" if t["ref_mean"] < t["comp_mean"] else "FULL worse"
            if t["metric"] in ["success", "spl", "recovery_rate", "min_clearance", "reaction_distance"]:
                direction = "FULL better" if t["ref_mean"] > t["comp_mean"] else "FULL worse"
            print(f"    {t['scenario']} | {t['comparison']} | {t['metric']}: "
                  f"p={t['p_adjusted']:.4f}, effect={t.get('effect_size', 0):.3f} ({direction})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis of ablation results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing ablation result JSON files")
    parser.add_argument("--output-dir", type=str, default="stats",
                        help="Directory for statistical output")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    #analyse all result files found
    for fname in sorted(os.listdir(args.results_dir)):
        if fname.endswith("_results.json"):
            path = os.path.join(args.results_dir, fname)
            print(f"\nAnalysing {fname}...")
            analyse_ablation(path, args.output_dir, verbose=not args.quiet)
