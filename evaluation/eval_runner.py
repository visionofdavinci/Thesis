"""
Evaluation Runner for Ablation Studies

Usage:
    python eval_runner.py --ablation ablation_* --ppo-checkpoint ./ppo_escape_v6_checkpoints --ppo-nav-checkpoint ./ppo_nav_checkpoints --n-episodes 200

Runs all (config, scenario) pairs for the specified ablation study, computes metrics, and saves results to JSON for statistical analysis.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from typing import Dict, List

from evaluation.eval_config import ABLATION_STUDIES, SYSTEM_CONFIGS, SCENARIO_FAMILIES, N_EPISODES, DT
from evaluation.eval_environment import EvalEnvironment
from evaluation.eval_metrics import compute_all_metrics, aggregate_metrics


def run_single_episode(system_config, scenario, ppo_checkpoint, seed, ppo_nav_checkpoint: str = ""):
    """
    Runs one episode and returns the metrics dict.
    PPO_ONLY uses ppo_nav_checkpoint (pure navigation policy);
    all other configs use ppo_checkpoint (escape/hybrid policy).
    """
    ckpt = (ppo_nav_checkpoint if system_config.name == "PPO_ONLY" and ppo_nav_checkpoint else ppo_checkpoint)
    env = EvalEnvironment(
        system_config=system_config,
        scenario=scenario,
        ppo_checkpoint=ckpt,
        seed=seed,
    )
    result = env.run_episode()
    metrics = compute_all_metrics(result, dt=DT)
    return metrics


def run_ablation(ablation_id: str, ppo_checkpoint: str, n_episodes: int = N_EPISODES, output_dir: str = "results", verbose: bool = True, ppo_nav_checkpoint: str = ""):
    """
    Runs a complete ablation study.

    For each (config, scenario) pair:
        - runs n_episodes with seeds 0..n_episodes-1
        - computes per-episode metrics
        - aggregates into summary statistics
        - saves raw + summary results to JSON

    Paired design: same seed = same obstacle jitter + start position,
    enabling paired statistical tests (Wilcoxon signed-rank).
    """
    if ablation_id not in ABLATION_STUDIES:
        raise ValueError(f"Unknown ablation: {ablation_id}. "
                         f"Options: {list(ABLATION_STUDIES.keys())}")

    study = ABLATION_STUDIES[ablation_id]
    scenarios = SCENARIO_FAMILIES[study.scenario_family]()
    configs = [SYSTEM_CONFIGS[name] for name in study.configs]

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print(f"ABLATION STUDY: {study.name}")
        print(f"  {study.description}")
        print(f"  Scenarios: {[s.name for s in scenarios]}")
        print(f"  Configs:   {[c.name for c in configs]}")
        print(f"  Episodes:  {n_episodes} per (config, scenario)")
        print(f"  RQs:       {study.rqs}")
        print(f"  Primary:   {study.primary_metrics}")
        print("=" * 70)

    all_results = {}

    for scenario in scenarios:
        for config in configs:
            key = f"{config.name}__{scenario.name}"
            if verbose:
                print(f"\n  Running {key} ({n_episodes} episodes)...", flush=True)

            episode_metrics = []
            t_start = time.time()

            for ep in range(n_episodes):
                metrics = run_single_episode(config, scenario, ppo_checkpoint, seed=ep, ppo_nav_checkpoint=ppo_nav_checkpoint)
                episode_metrics.append(metrics)

                if verbose and (ep + 1) % 50 == 0:
                    sr = sum(1 for m in episode_metrics if m["success"] == 1.0) / len(episode_metrics)
                    cr = sum(1 for m in episode_metrics if m["collision"] == 1.0) / len(episode_metrics)
                    elapsed = time.time() - t_start
                    print(f"    ep {ep+1}/{n_episodes}  SR={sr:.2f}  CR={cr:.2f}  "
                          f"({elapsed:.1f}s)", flush=True)

            #aggregate
            summary = aggregate_metrics(episode_metrics)
            elapsed = time.time() - t_start

            if verbose:
                sr = summary["success_rate"]["mean"]
                cr = summary["collision_rate"]["mean"]
                print(f"    Done: SR={sr:.3f}  CR={cr:.3f}  ({elapsed:.1f}s)")

            all_results[key] = {
                "config": config.name,
                "scenario": scenario.name,
                "n_episodes": n_episodes,
                "wall_time_seconds": elapsed,
                "episode_metrics": episode_metrics,
                "summary": _serialize_summary(summary),
            }

    #save results
    output_path = os.path.join(output_dir, f"{study.name}_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)

    if verbose:
        print(f"\n  Results saved to {output_path}")
        _print_summary_table(study, all_results)

    return all_results


def _serialize_summary(summary):
    """Convert summary dict to JSON-serializable format (drop raw value arrays)."""
    clean = {}
    for key, val in summary.items():
        if isinstance(val, dict):
            clean[key] = {k: v for k, v in val.items() if k != "values"}
        else:
            clean[key] = val
    return clean


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if np.isnan(obj) if isinstance(obj, float) else False:
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _print_summary_table(study, all_results):
    """Prints a compact summary table for the ablation study."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {study.name}")
    print(f"{'=' * 70}")

    #collect unique scenarios and configs
    scenarios = set()
    configs_seen = []
    for key in all_results:
        config_name, scenario_name = key.split("__")
        scenarios.add(scenario_name)
        if config_name not in configs_seen:
            configs_seen.append(config_name)

    for scenario_name in sorted(scenarios):
        print(f"\n  Scenario: {scenario_name}")
        header = f"  {'Config':<12}"
        for metric in study.primary_metrics[:4]:
            header += f"  {metric[:16]:>16}"
        header += f"  {'SR':>6}  {'CR':>6}"
        print(header)
        print("  " + "-" * len(header))

        for config_name in configs_seen:
            key = f"{config_name}__{scenario_name}"
            if key not in all_results:
                continue
            s = all_results[key]["summary"]
            row = f"  {config_name:<12}"
            for metric in study.primary_metrics[:4]:
                if metric in s and "mean" in s[metric]:
                    val = s[metric]["mean"]
                    row += f"  {val:>16.4f}" if val is not None else f"  {'N/A':>16}"
                elif metric == "success_rate" and metric in s:
                    row += f"  {s[metric]['mean']:>16.4f}"
                elif metric == "collision_rate" and metric in s:
                    row += f"  {s[metric]['mean']:>16.4f}"
                else:
                    row += f"  {'N/A':>16}"
            sr = s.get("success_rate", {}).get("mean", 0)
            cr = s.get("collision_rate", {}).get("mean", 0)
            row += f"  {sr:>6.3f}  {cr:>6.3f}"
            print(row)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation evaluation study")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=list(ABLATION_STUDIES.keys()),
                        help="Which ablation study to run")
    parser.add_argument("--ppo-checkpoint", type=str, default="ppo_escape_v9",
                        help="Path to PPO checkpoint used by hybrid configs (FULL, NO_SUB, etc.)")
    parser.add_argument("--ppo-nav-checkpoint", type=str, default="ppo_nav_v4/final",
                        help="Path to pure-navigation PPO checkpoint used by PPO_ONLY")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES,
                        help="Episodes per (config, scenario) pair")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for saving results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    run_ablation(
        ablation_id=args.ablation,
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_nav_checkpoint=args.ppo_nav_checkpoint,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
