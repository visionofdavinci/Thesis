"""
Ablation Table Generator

Builds readable comparison tables from ablation result JSON files.
One table per scenario: rows = configs, columns = metrics, cell = mean (std) [sig].
Outputs: plain text, CSV, and LaTeX.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# config and scenario ordering

#canonical ordering for config rows (FULL first as reference, then the
#one-component-removed ablations, then the single-component baselines)
CONFIG_ORDER = [
    "FULL",
    "NO_SUB",
    "NO_SUPER",
    "NO_PPO",
    "SUB_ONLY",
    "SUPER_ONLY",
    "PPO_ONLY",
]


# metric metadata

@dataclass
class MetricSpec:
    """
    Metadata for a single metric column:
        key             : key in episode_metrics dict
        label           : short header label
        better          : "high" means higher is better, "low" means lower
        kind            : "rate" (binary mean), "cont" (mean and std), "cont_no_std" (mean only), "count" (integer mean)
        precision       : decimal places in display
        latex_math_label: LaTeX math-mode label (optional; defaults to label)
        group           : RQ grouping for section dividers
    """
    key: str
    label: str
    better: str
    kind: str
    precision: int = 3
    latex_math_label: Optional[str] = None
    group: str = ""

    def latex_label(self) -> str:
        return self.latex_math_label or self.label


#metric specs per ablation - which columns appear in which table

METRICS_ABLATION_1 = [
    MetricSpec("log_dimensionless_jerk", "LDLJ",       "low",  "cont", 2, r"\mathrm{LDLJ}",   "RQ1 smoothness"),
    MetricSpec("velocity_variance",      "VelVar",     "low",  "cont", 3, r"\sigma^2_v",       "RQ1 smoothness"),
    MetricSpec("time_to_goal",           "TTG",        "low",  "cont", 2, r"t_{\mathrm{goal}}","RQ2 convergence"),
    MetricSpec("spl",                    "SPL",        "high", "cont", 3, r"\mathrm{SPL}",     "RQ2 convergence"),
    MetricSpec("path_length",            "PathLen",    "low",  "cont", 2, r"L",                "RQ2 convergence"),
    MetricSpec("success",                "SR",         "high", "rate", 3, r"\mathrm{SR}",      "overall"),
    MetricSpec("collision",              "CR",         "low",  "rate", 3, r"\mathrm{CR}",      "overall"),
]

METRICS_ABLATION_2 = [
    MetricSpec("collision",              "CR",         "low",  "rate", 3, r"\mathrm{CR}",      "RQ3 reactivity"),
    MetricSpec("min_clearance",          "MinClr",     "high", "cont", 3, r"d_{\min}",         "RQ3 reactivity"),
    MetricSpec("reaction_distance",      "ReactDist",  "high", "cont", 3, r"d_{\mathrm{rx}}",  "RQ3 reactivity"),
    MetricSpec("success",                "SR",         "high", "rate", 3, r"\mathrm{SR}",      "overall"),
    MetricSpec("time_to_goal",           "TTG",        "low",  "cont", 2, r"t_{\mathrm{goal}}","overall"),
    MetricSpec("detour_pct",             "Detour%",    "low",  "cont", 2, r"\Delta_{\%}",      "overall"),
]

METRICS_ABLATION_3 = [
    MetricSpec("success",                "SR",         "high", "rate", 3, r"\mathrm{SR}",      "RQ4 recovery"),
    MetricSpec("recovery_rate",          "Recov",      "high", "cont", 3, r"R_{\mathrm{rec}}", "RQ4 recovery"),
    MetricSpec("freeze_count",           "#Freeze",    "low",  "count",2, r"N_{\mathrm{fr}}",  "RQ4 recovery"),
    MetricSpec("detour_pct",             "Detour%",    "low",  "cont", 2, r"\Delta_{\%}",      "RQ4 recovery"),
    MetricSpec("collision",              "CR",         "low",  "rate", 3, r"\mathrm{CR}",      "overall"),
    MetricSpec("time_to_goal",           "TTG",        "low",  "cont", 2, r"t_{\mathrm{goal}}","overall"),
]

METRICS_ABLATION_4 = [
    MetricSpec("success",                "SR",         "high", "rate", 3, r"\mathrm{SR}",      "primary"),
    MetricSpec("collision",              "CR",         "low",  "rate", 3, r"\mathrm{CR}",      "primary"),
    MetricSpec("spl",                    "SPL",        "high", "cont", 3, r"\mathrm{SPL}",     "primary"),
    MetricSpec("log_dimensionless_jerk", "LDLJ",       "low",  "cont", 2, r"\mathrm{LDLJ}",    "primary"),
    MetricSpec("time_to_goal",           "TTG",        "low",  "cont", 2, r"t_{\mathrm{goal}}","secondary"),
    MetricSpec("velocity_variance",      "VelVar",     "low",  "cont", 3, r"\sigma^2_v",       "secondary"),
    MetricSpec("min_clearance",          "MinClr",     "high", "cont", 3, r"d_{\min}",         "secondary"),
    MetricSpec("recovery_rate",          "Recov",      "high", "cont", 3, r"R_{\mathrm{rec}}", "secondary"),
    MetricSpec("detour_pct",             "Detour%",    "low",  "cont", 2, r"\Delta_{\%}",      "secondary"),
]

METRICS_BY_ABLATION = {
    "ablation_1_subharmonic":      METRICS_ABLATION_1,
    "ablation_2_superharmonic":    METRICS_ABLATION_2,
    "ablation_3_ppo_escape":       METRICS_ABLATION_3,
    "ablation_4_full_integration": METRICS_ABLATION_4,
}


# aggregation helpers

def _mean_std(values: List[float]) -> Tuple[float, float, int]:
    """
    Mean and standard deviation over non-None, non-NaN values.
    Returns (mean, std, n_valid). Uses sample std (ddof=1) when n >= 2.
    Returns (nan, nan, 0) if nothing is valid.
    """
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(clean) / n
    if n == 1:
        return m, float("nan"), 1
    var = sum((v - m) ** 2 for v in clean) / (n - 1)
    return m, math.sqrt(var), n


def _rate(values: List[float]) -> Tuple[float, int]:
    """Mean over binary values, treating None/NaN as missing."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(clean)
    if n == 0:
        return float("nan"), 0
    return sum(clean) / n, n


# result ingestion

def _load_results(results_path: str) -> Dict[str, Dict]:
    with open(results_path, "r") as f:
        return json.load(f)


def _load_tests(tests_path: Optional[str]) -> List[Dict]:
    if tests_path is None or not os.path.exists(tests_path):
        return []
    with open(tests_path, "r") as f:
        return json.load(f)


def _scenario_order(results: Dict[str, Dict]) -> List[str]:
    """Stable scenario ordering: by scenario name (which encodes number)."""
    return sorted({v["scenario"] for v in results.values()})


def _configs_present(results: Dict[str, Dict], scenario: str) -> List[str]:
    """Configs actually present for this scenario, in canonical order."""
    present = {v["config"] for k, v in results.items() if v["scenario"] == scenario}
    return [c for c in CONFIG_ORDER if c in present]


def _episodes_for(results: Dict[str, Dict], config: str, scenario: str) -> List[Dict]:
    for entry in results.values():
        if entry["config"] == config and entry["scenario"] == scenario:
            return entry["episode_metrics"]
    return []


def _sig_lookup(tests: List[Dict], alpha: float = 0.05) -> Dict[Tuple[str, str, str, str], str]:
    """
    Build a dict mapping (scenario, reference, comparison, metric) -> sig marker.
    Marker is "*" if p_adjusted < alpha, "" otherwise, and "?" if NaN.
    """
    out: Dict[Tuple[str, str, str, str], str] = {}
    for t in tests:
        key = (t["scenario"], t["reference"], t["comparison"], t["metric"])
        p = t.get("p_adjusted")
        if p is None or (isinstance(p, float) and math.isnan(p)):
            out[key] = "?"  #test could not be performed (usually identical samples)
        elif p < alpha:
            out[key] = "*"
        else:
            out[key] = ""
    return out


# cell formatting

def _format_value(mean: float, std: float, spec: MetricSpec,
                  n_valid: int = -1, n_total: int = -1) -> str:
    """
    Numeric-only formatting of a metric value (no significance marker).

    When n_valid and n_total are supplied and n_valid < n_total, an
    `[n=X]` suffix is appended to make the denominator visible. This
    resolves the previous display ambiguity where `Recov 1.000` could
    mean either "1000/1000 episodes recovered" or "3/3 episodes that
    happened to freeze were recovered".
    """
    if math.isnan(mean):
        return "---"
    p = spec.precision
    if spec.kind == "rate":
        #rate metrics are already computed over all n_total episodes,
        #so no valid-count suffix is needed
        return f"{mean:.{p}f}"
    if spec.kind == "count":
        base = f"{mean:.{p}f}"
    elif spec.kind == "cont_no_std":
        base = f"{mean:.{p}f}"
    else:
        #cont: mean with std in parens
        if math.isnan(std):
            base = f"{mean:.{p}f}"
        else:
            base = f"{mean:.{p}f} ({std:.{p}f})"

    #append [n=X] only when a strict subset of episodes contributed
    if 0 <= n_valid < n_total and n_total > 0:
        base = f"{base} [n={n_valid}]"
    return base


def _cell(mean: float, std: float, spec: MetricSpec, sig: str,
          n_valid: int = -1, n_total: int = -1) -> str:
    """Full plain-text cell: value + trailing significance marker."""
    v = _format_value(mean, std, spec, n_valid=n_valid, n_total=n_total)
    if sig == "*":
        return v + "*"
    if sig == "?":
        return v + "·"   # mid-dot indicates "test not run / identical data"
    return v


# table builders

def build_rows(results: Dict[str, Dict], scenario: str, metric_specs: List[MetricSpec], tests: List[Dict], reference_config: str = "FULL", alpha: float = 0.05) -> Tuple[List[str], List[List[str]], List[List[float]], Dict]:
    """
    Build table rows for a single scenario.
    Returns:
        headers  : column headers (first = "Config", then one per metric)
        str_rows : 2D list of strings for display
        num_rows : 2D list of means (NaN where missing) for CSV export
        meta     : {'configs': [...], 'sig_matrix': [[...]], 'n_valid': [[...]], 'n_total': [[...]]}
    """
    sig_map = _sig_lookup(tests, alpha=alpha)
    configs = _configs_present(results, scenario)

    headers = ["Config"]
    for spec in metric_specs:
        arrow = "↑" if spec.better == "high" else "↓"
        headers.append(f"{spec.label} {arrow}")

    str_rows: List[List[str]] = []
    num_rows: List[List[float]] = []
    sig_matrix: List[List[str]] = []
    n_valid_rows: List[List[int]] = []
    n_total_rows: List[List[int]] = []

    for cfg in configs:
        eps = _episodes_for(results, cfg, scenario)
        n_total = len(eps)
        row_strs: List[str] = [cfg]
        row_nums: List[float] = []
        row_sigs: List[str] = []
        row_ns: List[int] = []
        row_totals: List[int] = []
        for spec in metric_specs:
            vals = [ep.get(spec.key) for ep in eps]
            if spec.kind == "rate":
                mean, n = _rate(vals)
                std = float("nan")
            else:
                mean, std, n = _mean_std(vals)
            sig = "" if cfg == reference_config else sig_map.get((scenario, reference_config, cfg, spec.key), "")
            #pass n_valid / n_total to the cell formatter so continuous
            #metrics with a reduced denominator get an [n=X] suffix
            row_strs.append(_cell(mean, std, spec, sig, n_valid=n, n_total=n_total))
            row_nums.append(mean)
            row_sigs.append(sig)
            row_ns.append(n)
            row_totals.append(n_total)
        str_rows.append(row_strs)
        num_rows.append(row_nums)
        sig_matrix.append(row_sigs)
        n_valid_rows.append(row_ns)
        n_total_rows.append(row_totals)

    meta = {
        "configs": configs,
        "sig_matrix": sig_matrix,
        "n_valid": n_valid_rows,
        "n_total": n_total_rows,
    }
    return headers, str_rows, num_rows, meta


# plain-text rendering

def render_text_table(headers: List[str], rows: List[List[str]], title: str = "") -> str:
    """Render a fixed-width plain-text table."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(fmt_row(headers))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


# CSV rendering (raw numeric + separate sig matrix + valid counts)

def write_csv(path: str, headers: List[str], num_rows: List[List[float]], configs: List[str], sig_matrix: List[List[str]], n_valid_rows: Optional[List[List[int]]] = None, n_total_rows: Optional[List[List[int]]] = None) -> None:
    """
    Write a CSV where each metric has *_mean, *_sig, and *_nvalid columns.
    The additional *_nvalid column exposes the valid-episodedenominator so downstream code can tell how trustworthy each cell is.
    """
    metric_headers = headers[1:]  #skip "Config"
    csv_headers = ["Config"]
    for mh in metric_headers:
        # mh is something like "SR ↑" - strip the arrow
        base = mh.split(" ")[0]
        csv_headers.append(f"{base}_mean")
        csv_headers.append(f"{base}_sig")
        csv_headers.append(f"{base}_nvalid")
        csv_headers.append(f"{base}_ntotal")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_headers)
        for i, (cfg, nums, sigs) in enumerate(zip(configs, num_rows, sig_matrix)):
            row = [cfg]
            row_nv = n_valid_rows[i] if n_valid_rows else [-1] * len(nums)
            row_nt = n_total_rows[i] if n_total_rows else [-1] * len(nums)
            for v, s, nv, nt in zip(nums, sigs, row_nv, row_nt):
                v_out = "" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6f}"
                row.append(v_out)
                row.append(s)
                row.append("" if nv < 0 else str(nv))
                row.append("" if nt < 0 else str(nt))
            w.writerow(row)


# LaTeX rendering

_LATEX_CONFIG_NAMES = {
    "FULL":       r"\textsc{Full}",
    "NO_SUB":     r"\textsc{No-Sub}",
    "NO_SUPER":   r"\textsc{No-Super}",
    "NO_PPO":     r"\textsc{No-Ppo}",
    "SUB_ONLY":   r"\textsc{Sub-Only}",
    "SUPER_ONLY": r"\textsc{Super-Only}",
    "PPO_ONLY":   r"\textsc{Ppo-Only}",
}


def _latex_cell(mean: float, std: float, spec: MetricSpec, sig: str, n_valid: int = -1, n_total: int = -1) -> str:
    """LaTeX cell with optional [n=X] subscript when the denominator shrinks."""
    if math.isnan(mean):
        return "---"
    p = spec.precision
    if spec.kind == "rate" or spec.kind == "count" or spec.kind == "cont_no_std":
        v = f"{mean:.{p}f}"
    elif math.isnan(std):
        v = f"{mean:.{p}f}"
    else:
        v = f"{mean:.{p}f}\\,({std:.{p}f})"
    #subscript the valid-episode count when it's a strict subset
    if (spec.kind != "rate" and 0 <= n_valid < n_total and n_total > 0):
        v = v + f"$_{{[n={n_valid}]}}$"
    if sig == "*":
        v = v + r"$^{*}$"
    elif sig == "?":
        v = v + r"$^{\cdot}$"
    return v


def render_latex_table(scenario: str, metric_specs: List[MetricSpec], configs: List[str], num_rows: List[List[float]], std_rows: List[List[float]], sig_matrix: List[List[str]],
                     caption: str, label: str, n_valid_rows: Optional[List[List[int]]] = None, n_total_rows: Optional[List[List[int]]] = None) -> str:
    """Render a booktabs table for inclusion in a LaTeX thesis."""
    col_spec = "l" + "r" * len(metric_specs)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header_cells = [r"Config"]
    for spec in metric_specs:
        arrow = r"$\uparrow$" if spec.better == "high" else r"$\downarrow$"
        header_cells.append(f"${spec.latex_label()}$ {arrow}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")
    for i, (cfg, means, stds, sigs) in enumerate(zip(configs, num_rows, std_rows, sig_matrix)):
        cells = [_LATEX_CONFIG_NAMES.get(cfg, cfg)]
        row_nv = n_valid_rows[i] if n_valid_rows else [-1] * len(means)
        row_nt = n_total_rows[i] if n_total_rows else [-1] * len(means)
        for mean, std, spec, sig, nv, nt in zip(means, stds, metric_specs, sigs, row_nv, row_nt):
            cells.append(_latex_cell(mean, std, spec, sig, n_valid=nv, n_total=nt))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_std_rows(results: Dict[str, Dict], scenario: str, metric_specs: List[MetricSpec]) -> Tuple[List[str], List[List[float]]]:
    """Parallel to build_rows but returns stds explicitly (for LaTeX)."""
    configs = _configs_present(results, scenario)
    stds = []
    for cfg in configs:
        eps = _episodes_for(results, cfg, scenario)
        row = []
        for spec in metric_specs:
            vals = [ep.get(spec.key) for ep in eps]
            if spec.kind == "rate":
                row.append(float("nan"))
            else:
                _, s, _ = _mean_std(vals)
                row.append(s)
        stds.append(row)
    return configs, stds


# per-ablation driver

@dataclass
class AblationProduct:
    """Container for all outputs of a single ablation analysis."""
    ablation_name: str
    text_path: str
    csv_paths: List[str]
    latex_path: str
    interpretation_path: str


def process_ablation(results_path: str, tests_path: Optional[str], output_dir: str, metric_specs: List[MetricSpec], ablation_label: str, ablation_caption_prefix: str, alpha: float = 0.05) -> AblationProduct:
    """
    Process one ablation end-to-end. Writes output files scoped to this ablation.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _load_results(results_path)
    tests = _load_tests(tests_path)

    scenarios = _scenario_order(results)
    ablation_name = os.path.basename(results_path).replace("_results.json", "")

    #plain text file: one table per scenario + header with methodology notes
    text_lines = []
    text_lines.append(f"ABLATION: {ablation_name}")
    text_lines.append("=" * (len(ablation_name) + 10))
    text_lines.append("")
    text_lines.append(f"Reference config: FULL (vs. each alternative, paired by episode seed)")
    text_lines.append(f"Significance:     * indicates FDR-adjusted p < {alpha} (Benjamini-Hochberg)")
    text_lines.append(f"                  · indicates paired test returned NaN (identical samples)")
    text_lines.append(f"Cells:            rate metrics = mean; continuous metrics = mean (std)")
    text_lines.append(f"                  [n=X] appears when only X of n_total episodes yielded a")
    text_lines.append(f"                  valid (non-NaN) value for that metric (e.g. TTG is only")
    text_lines.append(f"                  defined on successful episodes, Recov only on episodes")
    text_lines.append(f"                  that experienced at least one freeze event).")
    text_lines.append(f"Arrows:           ↑ higher is better, ↓ lower is better")
    text_lines.append("")

    csv_paths = []
    latex_tables = []

    for scenario in scenarios:
        headers, str_rows, num_rows, meta = build_rows( results, scenario, metric_specs, tests, alpha=alpha)
        configs, std_rows = build_std_rows(results, scenario, metric_specs)
        assert configs == meta["configs"], "config ordering mismatch"

        n_total_first = len(_episodes_for(results, configs[0], scenario))
        title = f"Scenario: {scenario}   (n_total = {n_total_first} episodes per config)"
        text_lines.append(render_text_table(headers, str_rows, title=title))
        text_lines.append("")

        #CSV (now includes _nvalid and _ntotal columns)
        csv_path = os.path.join(output_dir, f"{ablation_name}__{scenario}.csv")
        write_csv(csv_path, headers, num_rows, configs, meta["sig_matrix"], n_valid_rows=meta["n_valid"], n_total_rows=meta["n_total"])
        csv_paths.append(csv_path)

        #LaTeX (now includes [n=X] subscript when denominators shrink)
        caption = f"{ablation_caption_prefix}: scenario {scenario.replace('_', ' ')}"
        label = f"tab:{ablation_label}:{scenario}"
        latex_tables.append(render_latex_table(
            scenario=scenario,
            metric_specs=metric_specs,
            configs=configs,
            num_rows=num_rows,
            std_rows=std_rows,
            sig_matrix=meta["sig_matrix"],
            caption=caption,
            label=label,
            n_valid_rows=meta["n_valid"],
            n_total_rows=meta["n_total"],
        ))

    #write text
    text_path = os.path.join(output_dir, f"{ablation_name}_tables.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    #write latex
    latex_path = os.path.join(output_dir, f"{ablation_name}_tables.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(latex_tables))

    #interpretation stub (gets filled in by a separate module)
    interpretation_path = os.path.join(output_dir, f"{ablation_name}_interpretation.md")

    return AblationProduct(
        ablation_name=ablation_name,
        text_path=text_path,
        csv_paths=csv_paths,
        latex_path=latex_path,
        interpretation_path=interpretation_path,
    )
