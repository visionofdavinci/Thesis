"""
Evaluation Environment for Ablation Studies

Implements the full hybrid navigation loop (sub/super switching + PPO escape)
with configurable component ablation. This version of the file uses the
SAME production engines (SubharmonicFieldEngine, SuperharmonicFieldEngine,
PPOAgent, StuckDetector, build_escape_observation) that the training code
uses, so there can be no drift between "what we train" and "what we evaluate".

Navigation pipeline per step:
    1. compute field features (potential, gradient, dPhi/dt)
    2. compute current min clearance (for the proximity-based safety fallback)
    3. stuck detector checks for freeze state
    4. mode controller decides mode (sub / super / ppo), now taking
       min_clearance as a safety-override signal
    5. if sub/super: follow -gradient with dPhi/dt speed modulation
    6. if ppo: query PPO for velocity correction
    7. apply velocity, update obstacles, check termination
    8. log step data + per-step wall-clock time for the compute-cost metric
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from evaluation.eval_config import (
    SystemConfig, ScenarioConfig, ObstacleDef,
    FIELD_PARAMS, SUB_FIELD_PARAMS, MODE_CONTROLLER_PARAMS,
    STUCK_DETECTOR_PARAMS, NAV_PARAMS,
    GOAL_TOLERANCE, COLLISION_THRESHOLD, WALL_COLLISION_THRESHOLD,
    DT, MAX_ESCAPE_SPEED,
)

#production components -- these are what the training code uses
from engines.subharmonic_field_engine import SubharmonicFieldEngine, Obstacle
from engines.superharmonic_field_engine import SuperharmonicFieldEngine
from engines.rectangular_column import RectangularColumn
from engines.navigation_controller import (
    StuckDetector,
    ModeController as _ProductionModeController,
    build_escape_observation,
)
from engines.ppo_policy import PPOAgent
from engines.dqn_policy import DQNInferenceAgent

#eval-specific helpers for dropping columns into the analytical engines
from evaluation.environment_columns import (
    EvalColumnObstacle,
    subharmonic_potential_contribution,
    subharmonic_gradient_contribution,
    subharmonic_dphi_dt_contribution,
)


#column-aware wrappers around the production engines

class EvalSubEngineWithColumns(SubharmonicFieldEngine):
    """
    Thin eval-only subclass of the SubharmonicFieldEngine.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_obstacles: List[EvalColumnObstacle] = []

    def add_column(self, column_obstacle: EvalColumnObstacle) -> None:
        self.column_obstacles.append(column_obstacle)

    def compute_potential(self, pos: np.ndarray) -> float:
        phi = super().compute_potential(pos)
        #only contribute columns in analytical (subharmonic) mode -- grid
        #mode is 2D and has no sensible column representation
        if not self.subharmonic_mode:
            return phi
        for col_obs in self.column_obstacles:
            phi += subharmonic_potential_contribution(
                col_obs.column, pos, self.a_rep, self.k_rep)
        return phi

    def compute_gradient(self, pos: np.ndarray) -> np.ndarray:
        grad = super().compute_gradient(pos)
        if not self.subharmonic_mode:
            return grad
        for col_obs in self.column_obstacles:
            grad = grad + subharmonic_gradient_contribution(
                col_obs.column, pos, self.a_rep, self.k_rep)
        return grad

    def compute_dphi_dt(self, pos: np.ndarray) -> float:
        dphi = super().compute_dphi_dt(pos)
        if not self.subharmonic_mode:
            return dphi
        for col_obs in self.column_obstacles:
            dphi += subharmonic_dphi_dt_contribution(
                col_obs.column, pos, self.a_rep, self.k_rep)
        return dphi


class EvalSuperEngineWithColumns(SuperharmonicFieldEngine):
    """
    Eval-only subclass of the SuperharmonicFieldEngine with
    rectangular-column integration. Same design as EvalSubEngineWithColumns:
    columns are kept in a separate list so the superclass's sphere-only
    obstacle loop is unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_obstacles: List[EvalColumnObstacle] = []

    def add_column(self, column_obstacle: EvalColumnObstacle) -> None:
        self.column_obstacles.append(column_obstacle)

    def compute_potential(self, pos: np.ndarray) -> float:
        phi = super().compute_potential(pos)
        for col_obs in self.column_obstacles:
            phi += col_obs.column.compute_potential(
                pos, a_rep=self.a_rep, n_power=self.n_power,
                danger_distance=self.danger_distance, epsilon=self.epsilon,
            )
        return phi

    def compute_gradient(self, pos: np.ndarray) -> np.ndarray:
        grad = super().compute_gradient(pos)
        for col_obs in self.column_obstacles:
            grad = grad + col_obs.column.compute_gradient(
                pos, a_rep=self.a_rep, n_power=self.n_power,
                danger_distance=self.danger_distance, epsilon=self.epsilon,
            )
        return grad

    def compute_dphi_dt(self, pos: np.ndarray) -> float:
        dphi = super().compute_dphi_dt(pos)
        for col_obs in self.column_obstacles:
            dphi += col_obs.column.compute_dphi_dt(
                pos, a_rep=self.a_rep, n_power=self.n_power,
                danger_distance=self.danger_distance, epsilon=self.epsilon,
            )
        return dphi


#eval-specific mode controller (extends ModeController with a
#proximity-based safety fallback so sub mode can be held responsible for
#smoothness without also being responsible for close-range safety)

class EvalModeController(_ProductionModeController):
    """
    Extends the ModeController with a proximity-based safety
    fallback.
    """

    def __init__(self, high_threshold=0.1, low_threshold=0.03, min_dwell=3, proximity_threshold=0.35):
        super().__init__(high_threshold=high_threshold, low_threshold=low_threshold, min_dwell=min_dwell)
        self.proximity_threshold = proximity_threshold

    def update(self, dphi_dt: float, stuck: bool, min_clearance: float = float("inf")) -> str:
        self.steps_in_mode += 1
        proximity_trigger = min_clearance < self.proximity_threshold

        if stuck:
            if self.mode != "ppo":
                self.mode = "ppo"; self.steps_in_mode = 0; self.switch_count += 1
        elif self.mode == "ppo":
            if proximity_trigger or dphi_dt > self.high_threshold:
                self.mode = "superharmonic"
            else:
                self.mode = "subharmonic"
            self.steps_in_mode = 0; self.switch_count += 1
        elif proximity_trigger and self.mode != "superharmonic":
            #emergency override: skip dwell so super engages instantly
            self.mode = "superharmonic"; self.steps_in_mode = 0; self.switch_count += 1
        elif self.steps_in_mode >= self.min_dwell:
            if self.mode == "subharmonic" and dphi_dt > self.high_threshold:
                self.mode = "superharmonic"; self.steps_in_mode = 0; self.switch_count += 1
            elif (self.mode == "superharmonic"
                  and dphi_dt < self.low_threshold
                  and not proximity_trigger):
                self.mode = "subharmonic"; self.steps_in_mode = 0; self.switch_count += 1

        self.mode_history.append(self.mode)
        return self.mode

class NearObstacleStallDetector:
    """
    Detects the "oscillating near a static obstacle" failure mode that
    the StuckDetector misses on short timescales.
    """

    def __init__(self, proximity_band: float = 0.5,
                 speed_threshold: float = 0.25,
                 dwell_steps: int = 15):
        self.proximity_band = proximity_band
        self.speed_threshold = speed_threshold
        self.dwell_steps = dwell_steps
        self.stall_counter = 0
        self.is_stalled = False

    def update(self, velocity: np.ndarray, min_clearance: float) -> bool:
        speed = float(np.linalg.norm(velocity))
        is_near = min_clearance < self.proximity_band
        is_slow = speed < self.speed_threshold
        if is_near and is_slow:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        self.is_stalled = self.stall_counter >= self.dwell_steps
        return self.is_stalled

    def reset(self):
        self.stall_counter = 0
        self.is_stalled = False


#deterministic-inference wrappers for the two supported escape policies
#Both expose the same interface:
#   is_loaded: bool
#   select_action_deterministic(obs: np.ndarray) -> np.ndarray (3-dim direction)
#so EvalEnvironment._compute_ppo_velocity() works unchanged for either.

class _EvalPPOWrapper:
    """
    Loads a trained PPOAgent and exposes a deterministic action selector
    (mean of the pre-tanh Gaussian, then tanh) for reproducible evaluation.
    """

    def __init__(self, checkpoint_dir: str,
                 obs_dim: int = 13, act_dim: int = 3,
                 hidden_sizes=None):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        cfg_path = os.path.join(checkpoint_dir, "config.json") if checkpoint_dir else None
        if cfg_path and os.path.exists(cfg_path):
            import json
            with open(cfg_path) as _f:
                _cfg = json.load(_f)
            obs_dim = _cfg.get("obs_dim", obs_dim)
            act_dim = _cfg.get("act_dim", act_dim)
            hidden_sizes = _cfg.get("hidden_sizes", hidden_sizes)

        self.agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                              hidden_sizes=hidden_sizes)
        self._loaded = False
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            try:
                self.agent.load(checkpoint_dir)
                self._loaded = True
            except Exception as e:
                print(f"[eval] PPO checkpoint load failed ({e}); falling back "
                      "to random escape velocity for this evaluation run.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def select_action_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Mean action (no sampling) -> tanh squashed."""
        mean, _std = self.agent.policy.forward(obs)
        return np.tanh(mean)


class _EvalDQNWrapper:
    """
    Wraps DQNInferenceAgent with the same interface as _EvalPPOWrapper so
    EvalEnvironment._compute_ppo_velocity() works without any changes.

    Loads dqn_agent.pt + obs_normaliser.npz from checkpoint_dir.
    select_action_deterministic returns a unit direction from ACTION_MAP;
    multiplying by MAX_ESCAPE_SPEED gives the escape velocity.
    """

    def __init__(self, checkpoint_dir: str):
        self.agent = DQNInferenceAgent(checkpoint_dir)

    @property
    def is_loaded(self) -> bool:
        return self.agent.is_loaded

    def select_action_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Greedy DQN action -> unit direction vector (3-dim)."""
        return self.agent.select_action_deterministic(obs)


#step log entry

@dataclass
class StepLog:
    """Per-step data for metric computation."""
    position: np.ndarray
    velocity: np.ndarray
    mode: str
    potential: float
    dphi_dt: float
    d_goal: float
    min_clearance: float
    min_wall: float
    gradient_mag: float


#evaluation environment

class EvalEnvironment:
    """
    Full hybrid navigation environment for ablation evaluation.
    Configurable through SystemConfig to enable/disable components.
    """

    def __init__(self, system_config: SystemConfig,
                 scenario: ScenarioConfig,
                 ppo_checkpoint: str = "",
                 dqn_checkpoint: str = "",
                 seed: int = 0):
        self.cfg = system_config
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        #workspace bounds come from the scenario
        self.ws_lo = scenario.ws_lo
        self.ws_hi = scenario.ws_hi

        goal = scenario.goal.copy()

        #build production field engines
        self.sub_engine = EvalSubEngineWithColumns(
            x_bounds=(self.ws_lo, self.ws_hi),
            y_bounds=(self.ws_lo, self.ws_hi),
            grid_resolution=20,
            goal_position=goal,
        )
        #enable analytical mode with the paper's parameters
        self.sub_engine.subharmonic_mode = True
        self.sub_engine.a_att = SUB_FIELD_PARAMS["a_att"]
        self.sub_engine.a_rep = SUB_FIELD_PARAMS["a_rep"]
        self.sub_engine.k_rep = SUB_FIELD_PARAMS["k_rep"]
        self.sub_engine.danger_distance = float(np.sqrt(
            len(goal) / (2.0 * SUB_FIELD_PARAMS["k_rep"])))

        self.super_engine = EvalSuperEngineWithColumns(
            goal_position=goal,
            workspace_lo=self.ws_lo, workspace_hi=self.ws_hi,
            **FIELD_PARAMS,
        )

        #for PPO observation building we use the super engine
        self.obs_engine = self.super_engine

        #populate obstacles (spheres go into both engines through add_obstacle,
        #columns go into the column_obstacles side list). Keep a separate
        #list on self.obstacles for the step loop's bookkeeping.
        self.obstacles: List = []
        for odef in scenario.obstacles:
            if odef.kind == "column":
                jitter = self.rng.randn(2) * scenario.pos_jitter
                cxy = np.array([odef.position[0], odef.position[1]]) + jitter
                cxy = np.clip(cxy, self.ws_lo + 0.3, self.ws_hi - 0.3)
                vxy = None
                if odef.velocity is not None:
                    speed_scale = 1.0 + self.rng.uniform(
                        -scenario.speed_jitter, scenario.speed_jitter)
                    vxy = np.array([odef.velocity[0], odef.velocity[1]]) * speed_scale
                col = RectangularColumn(
                    center_xy=cxy,
                    half_extents_xy=odef.half_extents_xy.copy(),
                    z_lo=self.ws_lo, z_hi=self.ws_hi,
                    velocity_xy=vxy,
                    label="column",
                )
                eco = EvalColumnObstacle(column=col)
                self.obstacles.append(eco)
                #columns are added to BOTH engines' column-side list
                self.sub_engine.add_column(eco)
                self.super_engine.add_column(eco)
            else:
                pos = odef.position.copy() + self.rng.randn(3) * scenario.pos_jitter
                pos = np.clip(pos, self.ws_lo + 0.3, self.ws_hi - 0.3)
                vel = None
                if odef.velocity is not None:
                    speed_scale = 1.0 + self.rng.uniform(
                        -scenario.speed_jitter, scenario.speed_jitter)
                    vel = odef.velocity.copy() * speed_scale
                sphere = Obstacle(position=pos, radius=odef.radius, velocity=vel)
                self.obstacles.append(sphere)
                self.sub_engine.obstacles.append(sphere)
                self.super_engine.obstacles.append(sphere)

        #chaser indices
        self.chaser_idx = scenario.chaser_idx
        self.chaser_speed = scenario.chaser_speed

        self.mode_ctrl = EvalModeController(**MODE_CONTROLLER_PARAMS)
        self.stuck_det = StuckDetector(**STUCK_DETECTOR_PARAMS)
        self.near_obstacle_det = NearObstacleStallDetector(
            proximity_band=0.5,
            speed_threshold=0.25,
            dwell_steps=15,
        )

        #Escape agent -- PPO or DQN, both expose select_action_deterministic().
        #Priority: if dqn_checkpoint is given it takes precedence over ppo_checkpoint
        #so you can swap policies by just changing one argument.
        self.ppo_agent = None
        if system_config.use_ppo:
            if dqn_checkpoint:
                self.ppo_agent = _EvalDQNWrapper(dqn_checkpoint)
            elif ppo_checkpoint:
                self.ppo_agent = _EvalPPOWrapper(ppo_checkpoint)

        #start position with jitter, clipped into the workspace interior
        self.start_pos = scenario.start.copy() + self.rng.randn(3) * scenario.start_jitter
        self.start_pos = np.clip(self.start_pos, self.ws_lo + 0.3, self.ws_hi - 0.3)

        #state
        self.agent_pos = self.start_pos.copy()
        self.initial_d_goal = max(np.linalg.norm(self.agent_pos - goal), 1.0)
        self.step_count = 0
        self.done = False
        self.success = False
        self.collision = False
        self.wall_collision = False
        self.freeze_events = 0
        self.recovery_events = 0
        self._in_freeze = False
        self._last_velocity = np.zeros(3, dtype=float)

        #trajectory log
        self.step_log: List[StepLog] = []
        #per-step wall-clock timing for the computational-cost metric
        self._step_times: List[float] = []

    def _get_active_engine(self, mode: str):
        """Returns the field engine for the current mode."""
        if mode == "subharmonic":
            return self.sub_engine
        return self.super_engine

    def _compute_field_velocity(self, mode: str) -> np.ndarray:
        """Gradient-based velocity for sub/super modes with dPhi/dt modulation."""
        engine = self._get_active_engine(mode)
        gradient = engine.compute_gradient(self.agent_pos)
        dphi_dt = engine.compute_dphi_dt(self.agent_pos)

        #base velocity = -gradient (gradient descent), capped at max_velocity
        vel = -gradient
        vel_mag = np.linalg.norm(vel)
        if vel_mag > 1e-6:
            target_speed = min(vel_mag * NAV_PARAMS["velocity_gain"],
                                NAV_PARAMS["max_velocity"])
            vel = vel / vel_mag * target_speed
        else:
            vel = np.zeros(3)

        #dPhi/dt speed modulation: slow down when obstacle approaching
        speed_scale = 1.0 - NAV_PARAMS["dphi_dt_gain"] * max(dphi_dt, 0.0)
        speed_scale = max(speed_scale, NAV_PARAMS["min_speed_scale"])
        vel *= speed_scale

        return vel

    def _compute_ppo_velocity(self) -> np.ndarray:
        """PPO escape velocity."""
        if self.ppo_agent is None or not self.ppo_agent.is_loaded:
            #fallback: random perturbation (so PPO_ONLY without a checkpoint
            #does SOMETHING rather than freezing entirely)
            return self.rng.randn(3) * 0.5

        obs = build_escape_observation(
            self.agent_pos, self.obs_engine, self.step_count,
            max_episode_steps=self.scenario.max_steps,
            initial_d_goal=self.initial_d_goal,
            ws_lo=self.ws_lo, ws_hi=self.ws_hi,
        )
        action = self.ppo_agent.select_action_deterministic(obs)
        return action * MAX_ESCAPE_SPEED

    def _compute_min_clearance(self) -> float:
        """
        Surface-distance to nearest obstacle. Used both by the proximity- based mode-switch 
        override and by the collision check.
        """
        min_clr = float("inf")
        for obs in self.obstacles:
            if isinstance(obs, EvalColumnObstacle):
                d = obs.surface_distance(self.agent_pos)
            else:
                d = np.linalg.norm(self.agent_pos - obs.position) - obs.radius
            min_clr = min(min_clr, d)
        return min_clr

    def step(self) -> bool:
        """
        Execute one navigation step.
        Returns True if episode is done.
        """
        if self.done:
            return True

        #start wall-clock timer for the control-loop work of this step
        _t_step_start = time.perf_counter()

        #compute field features and proximity before deciding mode
        dphi_dt = self.super_engine.compute_dphi_dt(self.agent_pos)
        d_goal = np.linalg.norm(self.agent_pos - self.scenario.goal)
        current_min_clearance = self._compute_min_clearance()

        stuck_global = self.stuck_det.update(self.agent_pos, d_goal)
        stuck_local = self.near_obstacle_det.update(
            self._last_velocity, current_min_clearance,
        )
        stuck = stuck_global or stuck_local
        if stuck and not self._in_freeze:
            self.freeze_events += 1
            self._in_freeze = True
        elif not stuck and self._in_freeze:
            self.recovery_events += 1
            self._in_freeze = False

        #mode decision: PPO override only triggers if use_ppo is enabled AND stuck
        ppo_stuck = stuck and self.cfg.use_ppo
        if self.cfg.use_switching:
            #FULL and NO_PPO use the full switching logic (with proximity fallback)
            mode = self.mode_ctrl.update(dphi_dt, ppo_stuck,
                                          min_clearance=current_min_clearance)
        elif self.cfg.use_ppo and not self.cfg.use_sub and not self.cfg.use_super:
            #PPO_ONLY: always use PPO
            mode = "ppo"
        elif self.cfg.use_ppo and ppo_stuck:
            #configs with PPO but no switching: use PPO only when stuck
            mode = "ppo"
        elif self.cfg.use_sub and not self.cfg.use_super:
            mode = "subharmonic"
        elif self.cfg.use_super and not self.cfg.use_sub:
            mode = "superharmonic"
        else:
            mode = "subharmonic"

        #compute velocity based on mode
        if mode == "ppo":
            velocity = self._compute_ppo_velocity()
        else:
            velocity = self._compute_field_velocity(mode)

        #apply velocity, clip to workspace
        self.agent_pos = self.agent_pos + velocity * DT
        self.agent_pos = np.clip(self.agent_pos, self.ws_lo, self.ws_hi)
        self.step_count += 1
        #v4: cache this step's velocity for next-step stall detection
        self._last_velocity = velocity.copy()

        #chaser pursuit (if any)
        ci = self.chaser_idx
        if 0 <= ci < len(self.obstacles) and self.chaser_speed > 0:
            chaser = self.obstacles[ci]
            if not isinstance(chaser, EvalColumnObstacle):
                chase_vec = self.agent_pos - chaser.position
                cd = np.linalg.norm(chase_vec)
                if cd > 1e-6:
                    chaser.velocity = (chase_vec / cd) * self.chaser_speed

        #update obstacle positions (shared objects, so engines see the update)
        for i, obs in enumerate(self.obstacles):
            if isinstance(obs, EvalColumnObstacle):
                obs.update_position(DT, ws_lo=self.ws_lo, ws_hi=self.ws_hi)
            else:
                obs.update_position(DT)
                if i != ci and obs.velocity is not None:
                    #bounce spheres off the workspace walls
                    for d in range(3):
                        lo = self.ws_lo + obs.radius
                        hi = self.ws_hi - obs.radius
                        if obs.position[d] <= lo or obs.position[d] >= hi:
                            obs.velocity[d] *= -1
                            obs.position[d] = np.clip(obs.position[d], lo, hi)
                elif i == ci:
                    for d in range(3):
                        obs.position[d] = np.clip(obs.position[d],
                                                  self.ws_lo + obs.radius,
                                                  self.ws_hi - obs.radius)

        #collision and termination checks (recompute clearance post-move)
        d_goal = np.linalg.norm(self.agent_pos - self.scenario.goal)
        min_clearance = self._compute_min_clearance()
        collision = min_clearance < COLLISION_THRESHOLD

        wall_d = np.minimum(self.agent_pos - self.ws_lo, self.ws_hi - self.agent_pos)
        min_wall = float(wall_d.min())
        wall_collision = min_wall < WALL_COLLISION_THRESHOLD
        goal_reached = d_goal < GOAL_TOLERANCE

        #compute current field features for logging
        active_engine = self._get_active_engine(mode)
        potential = active_engine.compute_potential(self.agent_pos)
        gradient = active_engine.compute_gradient(self.agent_pos)

        #stop the per-step timer before the log write so we're timing the actual decision + physics work only
        _step_elapsed = time.perf_counter() - _t_step_start
        self._step_times.append(_step_elapsed)

        self.step_log.append(StepLog(
            position=self.agent_pos.copy(),
            velocity=velocity.copy(),
            mode=mode,
            potential=float(potential),
            dphi_dt=float(dphi_dt),
            d_goal=float(d_goal),
            min_clearance=float(min_clearance),
            min_wall=min_wall,
            gradient_mag=float(np.linalg.norm(gradient)),
        ))

        #termination
        if collision or wall_collision or goal_reached or self.step_count >= self.scenario.max_steps:
            self.done = True
            self.success = goal_reached
            self.collision = collision or wall_collision

        return self.done

    def run_episode(self) -> Dict:
        """Runs a full episode and returns summary results."""
        while not self.step():
            pass

        positions = np.array([s.position for s in self.step_log])
        velocities = np.array([s.velocity for s in self.step_log])

        if len(positions) > 1:
            path_length = float(sum(
                np.linalg.norm(positions[i+1] - positions[i])
                for i in range(len(positions) - 1)
            ))
        else:
            path_length = 0.0

        shortest_path = float(np.linalg.norm(self.start_pos - self.scenario.goal))

        mode_counts = {"subharmonic": 0, "superharmonic": 0, "ppo": 0}
        for s in self.step_log:
            if s.mode in mode_counts:
                mode_counts[s.mode] += 1

        return {
            "success": self.success,
            "collision": self.collision,
            "steps": self.step_count,
            "path_length": path_length,
            "shortest_path": shortest_path,
            "d_goal_final": self.step_log[-1].d_goal if self.step_log else float("inf"),
            "min_clearance_episode": min(s.min_clearance for s in self.step_log) if self.step_log else 0.0,
            "freeze_events": self.freeze_events,
            "recovery_events": self.recovery_events,
            "mode_switches": self.mode_ctrl.switch_count,
            "mode_counts": mode_counts,
            "positions": positions,
            "velocities": velocities,
            "modes": [s.mode for s in self.step_log],
            "d_goals": [s.d_goal for s in self.step_log],
            "potentials": [s.potential for s in self.step_log],
            "dphi_dts": [s.dphi_dt for s in self.step_log],
            "clearances": [s.min_clearance for s in self.step_log],
            "gradient_mags": [s.gradient_mag for s in self.step_log],
            #per-step wall-clock times (seconds) -- consumed by
            #eval_metrics.compute_all_metrics to produce computation_time
            "step_times": list(self._step_times),
        }