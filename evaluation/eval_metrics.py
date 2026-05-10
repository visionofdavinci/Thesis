"""
Metric Computation for Ablation Evaluation 

Computes all evaluation metrics from episode results.
Metrics are grouped by research question:

    RQ1 (smoothness):     log_dimensionless_jerk, velocity_variance, snap_integral
    RQ2 (convergence):    time_to_goal, spl, path_length
    RQ3 (reactivity):     collision_rate, min_clearance, reaction_distance
    RQ4 (freeze recovery): success_rate, recovery_rate, detour_pct, freeze_count
    Diagnostic:           mode_switches, computation_time
"""

import numpy as np
from typing import Dict, List

#numpy >= 2.0 renamed trapz to trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


#smoothness metrics (RQ1)

def compute_log_dimensionless_jerk(positions: np.ndarray, dt: float) -> float:
    """
    Log dimensionless jerk: scale-invariant smoothness measure.
    Lower values = smoother trajectories.

    Reference: Hogan & Sternad (2009) "Sensitivity of smoothness measures
    to movement duration, amplitude, and arrests"

    LDLJ = log(abs( (T^3 / L^2) * integral(||jerk||^2 dt) ))

    where T = duration, L = path length, jerk = d^3x/dt^3.
    Returns NaN for degenerate trajectories (< 4 points or zero path length).
    """
    if len(positions) < 4:
        return np.nan

    #velocities via central differences
    v = np.diff(positions, axis=0) / dt

    #accelerations
    a = np.diff(v, axis=0) / dt

    #jerk
    j = np.diff(a, axis=0) / dt

    if len(j) == 0:
        return np.nan

    #integral of ||jerk||^2
    jerk_sq = np.sum(j ** 2, axis=1)
    jerk_integral = _trapz(jerk_sq, dx=dt)

    #path length and duration
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    duration = len(positions) * dt

    if path_length < 1e-8 or duration < 1e-8:
        return np.nan

    #dimensionless jerk
    dj = (duration ** 3 / path_length ** 2) * jerk_integral

    return np.log(max(abs(dj), 1e-20))


def compute_velocity_variance(velocities: np.ndarray) -> float:
    """
    Variance of velocity magnitudes over the episode.
    High variance = oscillatory motion. Low variance = smooth, steady motion.
    """
    if len(velocities) < 2:
        return np.nan
    speed = np.linalg.norm(velocities, axis=1)
    return float(np.var(speed))


def compute_snap_integral(positions: np.ndarray, dt: float) -> float:
    """
    Snap integral: integral of ||d^4x/dt^4||^2 over the trajectory.
    Fourth derivative measures abruptness of acceleration changes.
    """
    if len(positions) < 5:
        return np.nan

    v = np.diff(positions, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    s = np.diff(j, axis=0) / dt

    if len(s) == 0:
        return np.nan

    snap_sq = np.sum(s ** 2, axis=1)
    return float(_trapz(snap_sq, dx=dt))


#convergence metrics (RQ2)

def compute_time_to_goal(steps: int, success: bool, dt: float) -> float:
    """
    Time to reach goal (seconds). Returns NaN for failed episodes.
    """
    if not success:
        return np.nan
    return steps * dt


def compute_spl(path_length: float, shortest_path: float, success: bool) -> float:
    """
    Success weighted by (normalised inverse) Path Length.
    SPL = success * shortest_path / max(path_length, shortest_path)

    Reference: Anderson et al. (2018) "On Evaluation of Embodied
    Navigation Agents"
    """
    if not success:
        return 0.0
    if shortest_path < 1e-8:
        return 1.0 if success else 0.0
    return shortest_path / max(path_length, shortest_path)


#safety metrics (RQ3)

def compute_min_clearance(clearances: List[float]) -> float:
    """Minimum obstacle clearance over the entire episode."""
    if not clearances:
        return np.nan
    return float(min(clearances))


def compute_reaction_distance(d_goals: List[float], dphi_dts: List[float], clearances: List[float], dphi_threshold: float = 0.1, min_valid_clearance: float = 0.05) -> float:
    """
    Distance to nearest obstacle when the system first reacts to a threat.
    "Reaction" = dPhi/dt exceeds threshold (obstacle approaching detected).
    Returns the clearance at the first reaction step.
    """
    for i, dphi in enumerate(dphi_dts):
        if dphi > dphi_threshold and clearances[i] >= min_valid_clearance:
            return clearances[i]
    return np.nan  #no reaction event detected above collision threshold


#freeze recovery metrics (RQ4)

def compute_recovery_rate(freeze_events: int, recovery_events: int) -> float:
    """
    Fraction of freeze events that were successfully recovered from.
    Returns NaN if no freeze events occurred.
    """
    if freeze_events == 0:
        return np.nan
    return recovery_events / freeze_events


def compute_detour_pct(path_length: float, shortest_path: float, success: bool = True) -> float:
    """
    Detour percentage: how much longer the actual path is vs straight line.
    detour_pct = (path_length - shortest_path) / shortest_path * 100

    Backward-compatible default (success=True) so callers that don't
    pass the flag see the old behaviour.
    """
    if not success:
        return np.nan
    if shortest_path < 1e-8:
        return 0.0
    return (path_length - shortest_path) / shortest_path * 100.0


#computational cost (diagnostic)

def compute_computation_time(step_times: List[float]) -> float:
    """
    Mean per-step wall-clock time in seconds. This measures the cost of
    one control-loop iteration (field evaluation + mode decision + PPO
    forward pass + physics + logging) averaged over the episode. It is
    the right number to compare FULL against SUB_ONLY / NO_PPO / etc.:
    FULL pays for both engines plus the controller plus the PPO forward
    pass, so its per-step cost should be the highest.

    Returns NaN if the list is empty.
    """
    if not step_times:
        return float("nan")
    return float(np.mean(step_times))


#aggregate helpers

def success_weighted_mean(values: List[float], successes: List[float]) -> float:
    """
    Mean of `values` over indices where `successes[i] == 1.0`.
    Returns NaN if no successes or all values are NaN.

    Useful for downstream analysis where a metric is only meaningful
    conditional on success (e.g. TTG, detour, SPL-on-successes).
    """
    if len(values) != len(successes):
        return float("nan")
    selected = [v for v, s in zip(values, successes) if s == 1.0 and v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not selected:
        return float("nan")
    return float(np.mean(selected))


#aggregate metric computation

def compute_all_metrics(result: Dict, dt: float = 0.1) -> Dict:
    """
    Computes all metrics from a single episode result dict.
    Returns a flat dict of metric_name -> value.
    """
    positions = result["positions"]
    velocities = result["velocities"]
    success_bool = bool(result["success"])

    metrics = {}

    #smoothness (RQ1)
    metrics["log_dimensionless_jerk"] = compute_log_dimensionless_jerk(positions, dt)
    metrics["velocity_variance"] = compute_velocity_variance(velocities)
    metrics["snap_integral"] = compute_snap_integral(positions, dt)

    #convergence (RQ2)
    metrics["time_to_goal"] = compute_time_to_goal(result["steps"], success_bool, dt)
    metrics["spl"] = compute_spl(result["path_length"], result["shortest_path"], success_bool)
    metrics["path_length"] = result["path_length"]

    #safety (RQ3)
    metrics["collision"] = 1.0 if result["collision"] else 0.0
    metrics["min_clearance"] = compute_min_clearance(result["clearances"])
    metrics["reaction_distance"] = compute_reaction_distance(
        result["d_goals"], result["dphi_dts"], result["clearances"],
    )

    #freeze recovery (RQ4)
    metrics["success"] = 1.0 if success_bool else 0.0
    metrics["recovery_rate"] = compute_recovery_rate(
        result["freeze_events"], result["recovery_events"],
    )
    #v4: detour is NaN for failed episodes (see compute_detour_pct docstring)
    metrics["detour_pct"] = compute_detour_pct(
        result["path_length"], result["shortest_path"], success=success_bool,
    )
    metrics["freeze_count"] = result["freeze_events"]

    #computational cost (diagnostic) -- handle absence gracefully so an
    #older result JSON without step_times still loads
    metrics["computation_time"] = compute_computation_time(
        result.get("step_times", [])
    )

    #diagnostic
    metrics["mode_switches"] = result["mode_switches"]
    metrics["steps"] = result["steps"]

    return metrics


def aggregate_metrics(all_episode_metrics: List[Dict]) -> Dict:
    """
    Aggregates per-episode metrics into summary statistics.
    Returns dict of metric_name -> {mean, std, median, values}.
    """
    if not all_episode_metrics:
        return {}

    keys = all_episode_metrics[0].keys()
    summary = {}

    for key in keys:
        values = [ep[key] for ep in all_episode_metrics]
        valid = [v for v in values if v is not None and not np.isnan(v)]

        summary[key] = {
            "mean": float(np.mean(valid)) if valid else np.nan,
            "std": float(np.std(valid)) if valid else np.nan,
            "median": float(np.median(valid)) if valid else np.nan,
            "n_valid": len(valid),
            "n_total": len(values),
            "values": values,  #keep raw for statistical tests
        }

    #convenience aliases
    n = len(all_episode_metrics)
    summary["success_rate"] = {
        "mean": sum(1 for ep in all_episode_metrics if ep["success"] == 1.0) / n,
        "n_total": n,
        "values": [ep["success"] for ep in all_episode_metrics],
    }
    summary["collision_rate"] = {
        "mean": sum(1 for ep in all_episode_metrics if ep["collision"] == 1.0) / n,
        "n_total": n,
        "values": [ep["collision"] for ep in all_episode_metrics],
    }

    return summary