"""
Ablation Study Configuration

scenarios are designed around four research questions:

    RQ1  oscillation reduction in narrow passages (with movers)
    RQ2  flat-region convergence efficiency near the goal
    RQ3  temporal reactivity to dynamic obstacles
    RQ4  PPO escape from 3D freeze-state traps

every scenario uses the full vertical range of the workspace so that
the drone is forced to plan in 3D. start and goal positions deliberately
have |Δz| ≥ 1.5 m in most scenarios; obstacles span z = 0.7 - 2.3 m.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


# system configurations (unchanged)

@dataclass
class SystemConfig:
    """Defines which architectural components are active."""
    name: str
    use_sub: bool
    use_super: bool
    use_switching: bool
    use_ppo: bool


SYSTEM_CONFIGS = {
    "FULL":       SystemConfig("FULL",       use_sub=True,  use_super=True,  use_switching=True,  use_ppo=True),
    "NO_SUB":     SystemConfig("NO_SUB",     use_sub=False, use_super=True,  use_switching=False, use_ppo=True),
    "NO_SUPER":   SystemConfig("NO_SUPER",   use_sub=True,  use_super=False, use_switching=False, use_ppo=True),
    "NO_PPO":     SystemConfig("NO_PPO",     use_sub=True,  use_super=True,  use_switching=True,  use_ppo=False),
    "SUB_ONLY":   SystemConfig("SUB_ONLY",   use_sub=True,  use_super=False, use_switching=False, use_ppo=False),
    "SUPER_ONLY": SystemConfig("SUPER_ONLY", use_sub=False, use_super=True,  use_switching=False, use_ppo=False),
    "PPO_ONLY":   SystemConfig("PPO_ONLY",   use_sub=False, use_super=False, use_switching=False, use_ppo=True),
}


# scenario definitions

@dataclass
class ObstacleDef:
    """
    Single obstacle specification before domain randomisation.

    For spheres: kind='sphere', position is (x,y,z), radius is the sphere radius.
    For columns: kind='column', position is (cx, cy, 0) [z ignored / spans workspace],
                 radius is half-diagonal of xy footprint (purely informational),
                 half_extents_xy gives the actual xy half-widths,
                 velocity is always None for columns (v4).
    """
    position: np.ndarray
    radius: float
    velocity: Optional[np.ndarray] = None
    kind: str = "sphere" #'sphere' or 'column'
    half_extents_xy: Optional[np.ndarray] = None #only used if kind == 'column'


@dataclass
class ScenarioConfig:
    """A single evaluation scenario."""
    name: str
    goal: np.ndarray
    start: np.ndarray
    obstacles: List[ObstacleDef]
    max_steps: int = 500
    chaser_idx: int = -1
    chaser_speed: float = 0.0
    ws_lo: float = -3.0
    ws_hi: float = 3.0
    pos_jitter: float = 0.25
    speed_jitter: float = 0.2
    start_jitter: float = 0.3


# helpers for building the scenarios

def _sphere(pos, r, vel=None):
    """Convenience: spherical obstacle. Motion allowed."""
    return ObstacleDef(position=np.array(pos, dtype=float), radius=float(r), velocity=np.array(vel, dtype=float) if vel is not None else None, kind="sphere")


def _column(center_xy, half_extents_xy):
    """
    Convenience: STATIC rectangular ceiling-to-floor column.

    columns are always static (motion is carried exclusively by spheres,
    per the system's design). the previous vel_xy argument has been
    removed to prevent accidental reintroduction of moving columns.
    """
    cx, cy = center_xy
    hx, hy = half_extents_xy
    radius = float(np.hypot(hx, hy)) # informational only
    return ObstacleDef(
        position=np.array([cx, cy, 0.0], dtype=float),
        radius=radius,
        velocity=None, # always static
        kind="column",
        half_extents_xy=np.array([hx, hy], dtype=float),
    )


# RQ1 + RQ2  - subharmonic contribution (smoothness + convergence)

def _make_scenarios_rq1_rq2() -> List[ScenarioConfig]:
    """
    RQ1 (oscillation reduction in narrow passages, dynamic obstacles) and RQ2 (flat-region convergence efficiency).

    each scenario uses a strong vertical separation between start and goal (|Δz| ≥ 2.0 m in two of the three scenarios) and distributes
    obstacles across the full workspace height, so the drone's path has  3D content rather than living in a near-planar slice.

    designed so SUB_ONLY excels (smooth path + reaches the corner-goal flat region), SUPER_ONLY oscillates in the narrow gate (RQ1) and
    drags through the long approach (RQ2), and FULL wins on the smoothness + convergence aggregate (logJerk, vel-variance, SPL, time-to-goal).
    """
    s1a = ScenarioConfig(
        name="S1a_vertical_chimney_with_passing_movers",
        # strong 3D start->goal: low corner -> high opposite corner
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # two columns perpendicular to the diagonal axis form a narrow
            # gate. opposing repulsion sources meet in the corridor and
            # this is the geometry that makes standard APF
            # oscillate (Borenstein & Koren, 1991). the subharmonic
            # field's smoother gradient should pass through cleanly.
            _column((-0.6,  0.6), (0.30, 0.30)),
            _column(( 0.6, -0.6), (0.30, 0.30)),
            # movers crossing the chimney at very different heights so
            # the drone has to commit to a vertical plan as well as a
            # lateral one. neither mover blocks a clean trajectory by
            # itself; together they force time-aware path shaping.
            _sphere([0.0, -1.6, 0.9], 0.25, [0.0,  0.10,  0.05]),
            _sphere([0.0,  1.6, 2.1], 0.25, [0.0, -0.10, -0.05]),
        ],
        max_steps=500,
    )

    s1b = ScenarioConfig(
        name="S1b_long_3d_approach_to_high_corner",
        # the longest realistic diagonal in the workspace - the goal sits
        # in a deliberately empty corner so the harmonic gradient flattens
        # out near it. this is the RQ2 flat-region test.
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.4, 2.4, 2.4]),
        obstacles=[
            # sparse mid-path clutter spread through the full vertical
            # range. deliberately none in the goal region so that the
            # final approach is dominated by the attractive gradient
            # alone (which is what flattens for harmonic fields).
            _sphere([-1.4, -0.6, 0.8], 0.30, None),  # low
            _sphere([-0.4, -1.4, 1.5], 0.30, None),  # mid
            _sphere([ 0.4,  0.4, 2.1], 0.30, [0.04, -0.04, 0.0]),  # high (slow drift)
        ],
        max_steps=500,
    )

    s1c = ScenarioConfig(
        name="S1c_3d_slalom_with_tangent_mover",
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # I had spheres essentially ON the diagonal path (one had
            # negative surface clearance), so the SUB engine collided
            # because its Gaussian repulsion peaks at d* = 1/sqrt(2k_rep)
            # = 0.71 m and decreases at smaller d. spheres are now offset
            # both laterally and vertically - vertical alternation
            # preserved (the original design intent), but with surface
            # clearance >= 0.65 m from the natural straight path so SUB's
            # smooth gradient can guide the detour without proximity
            # override having to fire.
            _sphere([-2.0, -0.8, 1.7], 0.30, None),  # offset+above (path z~1.0)
            _sphere([-0.1, -1.3, 0.7], 0.30, None),  # offset+below (path z~1.3)
            _sphere([-0.4,  0.8, 2.4], 0.30, None),  # offset+above (path z~1.7)
            _sphere([ 1.7,  0.5, 1.4], 0.30, None),  # offset+below (path z~2.1)
            # tangent mover drifts orthogonal to the path at mid-height
            _sphere([-1.0, 1.0, 1.5], 0.25, [0.10, -0.05, 0.0]),
        ],
        max_steps=500,
        # tighter than default (0.25) - the clearance-from-path budget
        # for SUB is small enough that worst-case jitter could still
        # push spheres into the danger zone otherwise
        pos_jitter=0.15,
    )

    return [s1a, s1b, s1c]


# RQ3  -  superharmonic contribution (temporal reactivity)

def _make_scenarios_rq3() -> List[ScenarioConfig]:
    """
    RQ3 (temporal reactivity to dynamic obstacles).

    each scenario contains a fast-approaching threat where the dPhi/dt signal rises well before the drone enters proximity range. sub mode
    only triggers a meaningful response when the threat is geometrically close,  super mode reacts when the rate of approach is high. so
    SUB_ONLY collides where SUPER_ONLY and FULL succeed.

    each test attacks the drone from a different threat class to make sure the result generalises:
        S2a -- vertical drop  (z-axis threat from above)
        S2b -- horizontal cross (xy-plane perpendicular intercept)
        S2c -- active pursuer (chaser tracks the drone in 3D)
    """
    s2a = ScenarioConfig(
        name="S2a_staggered_high_speed_descenders",
        # drone path is roughly horizontal (small dz) to isolate the
        # vertical threat axis. previously goal was at (2.5, 2.5, 1.6)
        # which put it INSIDE the SUPER engine's wall_danger zone for
        # both x=3 and y=3 walls (each only 0.5 m away vs. wall_danger
        # = 0.8 m). the wall potential created a barrier RIGHT at the
        # goal that SUPER_ONLY and NO_SUB couldn't push through, hence
        # those rows were 0.000 / 0.012 SR. moved 0.5 m inward.
        start=np.array([-2.5, -2.5, 1.4]),
        goal=np.array([2.0, 2.0, 1.8]),
        obstacles=[
            # benign clutter (off the natural path)
            _sphere([-1.5, -0.3, 1.2], 0.25, None),
            _sphere([ 0.3,  1.5, 1.8], 0.25, None),
            # TWO fast descenders timed to intercept different points
            # along the drone's path. v5 had a single slow drop (vz=-0.6)
            # which SUB handled trivially via proximity override - the
            # result was SUB_ONLY (0.969) == FULL (0.973), i.e. the
            # superharmonic component showed zero contribution. with two
            # threats at different intercept times the drone has to
            # react TWICE, and the lead-time advantage of dPhi/dt
            # actually matters.
            _sphere([-1.2, -1.2, 2.6], 0.30, [0.0, 0.0, -0.85]),  # ~step 13
            _sphere([ 0.2,  0.2, 2.6], 0.30, [0.0, 0.0, -0.40]),  # ~step 28
        ],
        max_steps=500,
    )

    s2b = ScenarioConfig(
        name="S2b_high_speed_perpendicular_crosser",
        start=np.array([-2.5, -2.5, 0.6]),
        goal=np.array([2.0, 2.0, 2.4]),
        obstacles=[
            # path-side static clutter spread vertically
            _sphere([-1.5, -0.5, 1.0], 0.25, None),
            _sphere([ 0.5,  1.5, 2.0], 0.25, None),
            # crosser starts off-diagonal and travels perpendicular to
            # the drone's natural diagonal at 0.85 m/s, intercepting
            # near the path midpoint at z = 1.5.
            _sphere([1.5, -1.5, 1.5], 0.30, [-0.6, 0.6, 0.0]),
        ],
        max_steps=500,
    )

    s2c = ScenarioConfig(
        name="S2c_dynamic_3d_clutter",
        start=np.array([-2.5, -2.5, 0.6]),
        goal=np.array([2.0, 2.0, 2.4]),
        obstacles=[
            # the chaser was placed at (-2.0, -2.0, 1.5) which is
            # only 1.14 m from the start - with chaser_speed=0.85 it
            # caught the drone before the drone had cleared the start
            # jitter envelope (CR = 1.000 across every config). chaser
            # removed entirely; replaced with sub-friendly column gate
            # + four crossing movers approaching from different
            # directions and altitudes. this still tests temporal
            # reactivity (movers) AND smoothness in narrow passages
            # (gate) but isn't dominated by a single broken setup.
            _column((-0.8,  0.8), (0.25, 0.25)),
            _column(( 0.8, -0.8), (0.25, 0.25)),
            # head-on low approaching from the east
            _sphere([2.0, 0.0, 0.9], 0.25, [-0.40, 0.0, 0.05]),
            # head-on mid approaching from the north
            _sphere([0.0, 2.0, 1.7], 0.25, [0.0, -0.45, 0.0]),
            # diagonal cross from NW
            _sphere([-1.5, 1.5, 1.3], 0.25, [0.40, -0.30, 0.0]),
            # diagonal cross from SE high
            _sphere([1.5, -1.5, 2.1], 0.30, [-0.30, 0.40, 0.0]),
        ],
        max_steps=500,
    )

    return [s2a, s2b, s2c]


# RQ4  -  PPO escape (3D static traps)

def _make_scenarios_rq4() -> List[ScenarioConfig]:
    """
    RQ4 (PPO escape from freeze states).

    each scenario has a static configuration that creates a true 3D local minimum / freeze state. pure analytical methods (sub, super,
    sub+super switching without PPO) get trapped, only configs with use_ppo=True can recover.

    the traps are designed so vertical escape alone doesn't dissolve them - either a ceiling sphere blocks z+ or the geometry creates
    a near-stationary point with no obvious escape direction. this is the main upgrade over the traps, which were 2D U-shapes the
    drone could climb over.
    """
    s3a = ScenarioConfig(
        name="S3a_centerline_blocker_3d",
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # one large sphere on the start-goal line at the midpoint.
            # the opposing attractive (toward goal) and repulsive (away
            # from sphere) gradients almost cancel along the line,
            # creating a near-stationary point. small jitter keeps the
            # symmetry breakable in principle, but standard APF rarely
            # escapes within 500 steps.
            _sphere([-0.25, -0.25, 1.5], 0.55, None),
        ],
        max_steps=500,
        pos_jitter=0.05,    # tight: keep the trap symmetric across seeds
        start_jitter=0.15,
    )

    s3b = ScenarioConfig(
        name="S3b_3d_pocket_trap",
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # a half-shell facing the start direction. back wall blocks
            # forward motion, side walls block lateral motion, ceiling
            # sphere blocks the z+ escape that defeated the previous
            # 2D U-traps. the only escape is z- or back along start
            # direction - both against the attractive gradient.
            _sphere([0.4, 0.4, 1.5], 0.35, None),    # back wall
            _sphere([0.7, 0.0, 1.5], 0.30, None),    # right wall
            _sphere([0.0, 0.7, 1.5], 0.30, None),    # left wall
            _sphere([0.3, 0.3, 2.1], 0.30, None),    # ceiling
        ],
        max_steps=500,
        pos_jitter=0.10,
        start_jitter=0.15,
    )

    s3c = ScenarioConfig(
        name="S3c_3d_double_cluster_saddle",
        start=np.array([-2.5, -2.5, 0.8]),
        goal=np.array([2.0, 2.0, 2.2]),
        obstacles=[
            # two clusters either side of the start->goal line create a
            # saddle - the attractive gradient bisects them and the
            # drone has to commit to one side. without a learned
            # escape, it tends to oscillate near the saddle point.
            _sphere([-0.2, 0.5, 1.5], 0.30, None),
            _sphere([ 0.0, 0.8, 1.5], 0.30, None),
            _sphere([ 0.5,-0.2, 1.5], 0.30, None),
            _sphere([ 0.8, 0.0, 1.5], 0.30, None),
            _sphere([ 0.2, 0.2, 2.4], 0.30, None),
        ],
        max_steps=500,
        pos_jitter=0.10,
        start_jitter=0.15,
    )

    return [s3a, s3b, s3c]


# FULL  -  integration stress test

def _make_scenarios_full() -> List[ScenarioConfig]:
    """
    Integration stress tests. each scenario combines smoothness, reactivity, and freeze-recovery challenges so that no single ablation can win on
    the aggregate.

    difficulty was deliberately reduced from the v4 version: the previous S4 scenarios were so dense that even FULL reached 0% SR, which
    destroyed the discrimination between configs. these two scenarios target ~50% SR for FULL, with monotone degradation as each
    component is removed.
    """
    s4a = ScenarioConfig(
        name="S4a_3d_traversal_with_late_threat",
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # sub-friendly static clutter on the early/middle path
            _sphere([-1.0, -0.8, 0.9], 0.25, None),
            _sphere([ 0.5,  0.0, 1.6], 0.25, None),
            # one late-stage mover near the goal - tests reactivity
            # exactly when sub mode should be engaged for smoothness
            _sphere([ 1.5,  0.5, 2.1], 0.25, [0.0, 0.30, 0.05]),
        ],
        max_steps=500,
    )

    s4b = ScenarioConfig(
        name="S4b_corridor_movers_pocket",
        start=np.array([-2.5, -2.5, 0.5]),
        goal=np.array([2.0, 2.0, 2.5]),
        obstacles=[
            # had 4 obstacles where the late static at (1.6, 1.6, 2.3)
            # had NEGATIVE surface clearance from the path (i.e. the
            # straight start->goal line passed through the sphere) AND
            # was only 0.5 m from the goal - so even FULL collided 44%
            # of the time. redesign:
            #   - 6 obstacles total (user's request)
            #   - column gate widened so each column is 1.13 m
            #     perpendicular from the path (was 0.85 m)
            #   - 3 crossing movers in middle/late zone for super test
            #   - 1 static OFFSET from path, kept >= 1.5 m from goal so
            #     drone has space to settle into goal-approach mode
            # ablation 4 should differentiate FULL from each ablation
            # without being so hard that FULL itself fails.

            # phase 1 (sub): diagonal column gate at midpoint
            _column((-0.8,  0.8), (0.25, 0.25)),
            _column(( 0.8, -0.8), (0.25, 0.25)),
            # phase 2 (super): two crossing movers in middle zone
            _sphere([-1.5, 1.5, 1.5], 0.25, [0.25, -0.25, 0.0]),
            _sphere([ 1.5, -1.5, 1.7], 0.25, [-0.25, 0.25, 0.05]),
            # phase 3 (super, late): slow descending mover ABOVE the
            # goal-approach corridor. starts laterally offset from path
            # (~1.0 m) and drifts toward it - the drone enters its
            # threat zone right as it's settling into goal-approach
            # mode. had this at (1.0, 1.5, 2.0) with surface_clr =
            # 0.14 m from the path which guaranteed FULL collisions.
            _sphere([2.0, 0.6, 2.5], 0.25, [0.0, 0.10, -0.05]),
            # phase 4 (PPO): off-axis static near goal approach.
            # placed at (0.3, 2.0, 1.8) -- 1.24 m perpendicular from
            # the natural path, 1.97 m from goal so the drone has room
            # to settle into goal-approach mode after the detour.
            _sphere([0.3, 2.0, 1.8], 0.25, None),
        ],
        max_steps=500,
    )

    return [s4a, s4b]


SCENARIO_FAMILIES = {
    "RQ1_RQ2": _make_scenarios_rq1_rq2,
    "RQ3":     _make_scenarios_rq3,
    "RQ4":     _make_scenarios_rq4,
    "FULL":    _make_scenarios_full,
}


# ablation study definitions

@dataclass
class AblationStudy:
    """Links configs, scenarios, metrics, and target research questions."""
    name: str
    description: str
    scenario_family: str
    configs: List[str]
    primary_metrics: List[str]
    secondary_metrics: List[str]
    rqs: List[str]


ABLATION_STUDIES = {
    "ablation_1": AblationStudy(
        name="ablation_1_subharmonic",
        description="Subharmonic contribution: smoothness (RQ1) and convergence (RQ2)",
        scenario_family="RQ1_RQ2",
        configs=["FULL", "NO_SUB", "NO_SUPER", "NO_PPO", "SUB_ONLY", "SUPER_ONLY", "PPO_ONLY"],
        primary_metrics=[
            "log_dimensionless_jerk",
            "velocity_variance",
            "time_to_goal",
            "spl",
        ],
        secondary_metrics=["success_rate", "collision_rate", "path_length", "computation_time"],
        rqs=["RQ1", "RQ2"],
    ),
    "ablation_2": AblationStudy(
        name="ablation_2_superharmonic",
        description="Superharmonic contribution: temporal reactivity (RQ3)",
        scenario_family="RQ3",
        configs=["FULL", "NO_SUB", "NO_SUPER", "NO_PPO", "SUB_ONLY", "SUPER_ONLY", "PPO_ONLY"],
        primary_metrics=[
            "collision_rate",
            "min_clearance",
            "reaction_distance",
        ],
        secondary_metrics=["success_rate", "time_to_goal", "detour_pct", "computation_time"],
        rqs=["RQ3"],
    ),
    "ablation_3": AblationStudy(
        name="ablation_3_ppo_escape",
        description="PPO escape contribution: freeze-state recovery (RQ4)",
        scenario_family="RQ4",
        configs=["FULL", "NO_SUB", "NO_SUPER", "NO_PPO", "SUB_ONLY", "SUPER_ONLY", "PPO_ONLY"],
        primary_metrics=[
            "success_rate",
            "recovery_rate",
            "detour_pct",
        ],
        secondary_metrics=["collision_rate", "time_to_goal", "freeze_count", "computation_time"],
        rqs=["RQ4"],
    ),
    "ablation_4": AblationStudy(
        name="ablation_4_full_integration",
        description="Full integration stress test (all RQs)",
        scenario_family="FULL",
        configs=["FULL", "NO_SUB", "NO_SUPER", "NO_PPO", "SUB_ONLY", "SUPER_ONLY", "PPO_ONLY"],
        primary_metrics=[
            "success_rate",
            "collision_rate",
            "spl",
            "log_dimensionless_jerk",
        ],
        secondary_metrics=[
            "time_to_goal", "velocity_variance", "min_clearance",
            "recovery_rate", "detour_pct", "computation_time",
        ],
        rqs=["RQ1", "RQ2", "RQ3", "RQ4"],
    ),
}


# shared constants

N_EPISODES = 1000
GOAL_TOLERANCE = 0.35
COLLISION_THRESHOLD = 0.05
WALL_COLLISION_THRESHOLD = 0.02
DT = 0.1
MAX_ESCAPE_SPEED = 1.5

# field engine parameters (match training)
FIELD_PARAMS = {
    "a_att": 0.1,
    "a_rep": 1.0,
    "n_power": 2.0,
    "danger_distance": 1.0,
    "a_wall": 0.5,
    "wall_power": 2.0,
    "wall_danger": 0.8,
}

SUB_FIELD_PARAMS = {
    "a_att": 0.1,
    "a_rep": 1.0,
    "k_rep": 1.0,
}

MODE_CONTROLLER_PARAMS = {
    "high_threshold": 0.1,
    "low_threshold": 0.03,
    "min_dwell": 3,
    "proximity_threshold": 0.25,
}

STUCK_DETECTOR_PARAMS = {
    "window_size": 30,
    "progress_threshold": 0.15,
    "cycle_ratio": 5.0,
    "stagnation_window": 25,
    "stagnation_threshold": 0.10,
}

NAV_PARAMS = {
    "velocity_gain": 2.0,
    "max_velocity": 1.5,
    "dphi_dt_gain": 0.4,
    "min_speed_scale": 0.2,
}