"""
PPO Escape Training (v9: column-aware)
"""

import argparse
import numpy as np
import os
from typing import Tuple

from engines.superharmonic_field_engine import SuperharmonicFieldEngine
from engines.ppo_policy import PPOAgent
from rectangular_column import RectangularColumn


def _obstacle_surface_distance(obstacle, agent_pos: np.ndarray) -> float:
    """
    Distance from agent to the nearest surface point of an obstacle.

    Uses obstacle.surface_distance(agent_pos) when available (RectangularColumn returns the true axis-aligned-box SDF), otherwise
    falls back to the spherical formula
        ||agent_pos - obstacle.position|| - obstacle.radius
    which is exact for spherical obstacles.
    """
    if hasattr(obstacle, 'surface_distance'):
        return float(obstacle.surface_distance(agent_pos))
    return float(np.linalg.norm(agent_pos - obstacle.position) - obstacle.radius)


#running statistics for return normalisation 

class RunningMeanStd:
    """
    Welford's online algorithm for running mean and variance.
    Used to normalise value targets for stable critic learning.

    Reference: Andrychowicz et al. (2021) "What Matters In On-Policy RL?"
    - return normalisation was the most impactful PPO implementation
    detail in their ablation study.

    Parameters:
    - shape : tuple - shape of the statistic to track
    - epsilon : float - minimum std to prevent division by zero
    """

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray):
        """
        Update running statistics with a batch of values.

        Parameters:
        - batch : np.ndarray - array of new values
        """
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalise values using running statistics.

        Parameters:
        - x : np.ndarray - values to normalise
        Returns:
        - normalised : np.ndarray
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# PBRS reward shaper with clipping 

class PBRSRewardShaper:
    """
    Dynamic Potential-Based Reward Shaping with bounded output.
    F(s,t,s',t') = clip(Phi(s,t) - gamma * Phi(s',t'), -C, +C)

    Changes:
      - potential_scale reduced from 10.0 to 1.0 
      - output clipped to [-clip_bound, +clip_bound] 

    Parameters:
    - field_engine : engine with compute_potential
    - gamma : float - discount factor (must match PPO gamma)
    - potential_scale : float - multiplier for Phi (now 1.0)
    - clip_bound : float - max absolute shaping reward per step
    """

    def __init__(self, field_engine, gamma: float = 0.99, potential_scale: float = 1.0, clip_bound: float = 5.0):
        self.engine = field_engine
        self.gamma = gamma
        self.potential_scale = potential_scale
        self.clip_bound = clip_bound
        self._prev_potential = None

    def reset(self, pos: np.ndarray):
        self._prev_potential = self.engine.compute_potential(pos) * self.potential_scale

    def compute_shaped_reward(self, pos_new: np.ndarray, r_task: float, done: bool) -> Tuple[float, float, float]:
        if done:
            phi_new = 0.0
        else:
            phi_new = self.engine.compute_potential(pos_new) * self.potential_scale

        f_shaping = self._prev_potential - self.gamma * phi_new

        #clip shaping to prevent magnitude explosion near walls
        f_shaping = np.clip(f_shaping, -self.clip_bound, self.clip_bound)

        r_shaped = r_task + f_shaping
        self._prev_potential = phi_new
        return r_shaped, r_task, f_shaping


class CurriculumManager:
    """
    Progressive difficulty curriculum for escape training.

    Parameters:
    - promotion_threshold : float - success rate to advance (stages 3+)
    - early_promotion_threshold : float - success rate for stages 0-2
    - window_size : int - rolling window for success rate
    - min_episodes_per_stage : int - minimum episodes before promotion
    """
    N_STAGES = 14

    def __init__(self, promotion_threshold: float = 0.45, early_promotion_threshold: float = 0.35, window_size: int = 50, min_episodes_per_stage: int = 60):
        self.promotion_threshold = promotion_threshold
        self.early_promotion_threshold = early_promotion_threshold
        self.window_size = window_size
        self.min_episodes = min_episodes_per_stage

        self.current_stage = 0
        self.episode_results = []
        self.episodes_in_stage = 0
        self.stage_history = []

        self._stage_names = [
            'Stage 0: goal-reaching (bootstrap)',
            'Stage 1: static blocker',           
            'Stage 2: wall push',                         
            'Stage 3: corner trap',                       
            'Stage 4: static U-trap',            
            'Stage 5: 2 close movers',                    
            'Stage 6: 4 mixed movers',                    
            'Stage 7: static saddle / cluster',  
            'Stage 8: column slalom',
            'Stage 9: 6 dense movers',                    
            'Stage 10: 3 movers + slow chaser',            
            'Stage 11: 5 movers + fast chaser',           
            'Stage 12: 6 movers + fast chaser',          
            'Stage 13: final approach (close start)',     
        ]
        # max-steps per stage. Static stages get more time (250-300) because
        # escape requires off-axis exploration; the analytical attractor
        # alone won't get the policy out of these. Movers / dense stages
        # keep their original step budgets. Stage 8 (column slalom) needs
        # off-axis exploration and gets 300 steps.
        self._max_steps = [
            200,  # 0  bootstrap
            250,  # 1  static blocker 
            150,  # 2  wall push
            200,  # 3  corner trap
            300,  # 4  static U-trap 
            200,  # 5  2 close movers
            250,  # 6  4 mixed
            300,  # 7  static saddle 
            300,  # 8  column slalom  
            300,  # 9  6 dense
            300,  # 10 3 + slow chaser
            350,  # 11 5 + fast
            400,  # 12 6 + fast
            300,  # 13 final approach
        ]
        self._chaser_speeds = [
            0.0,  # 0
            0.0,  # 1 static
            0.0,  # 2
            0.0,  # 3
            0.0,  # 4 static
            0.0,  # 5
            0.0,  # 6
            0.0,  # 7 static
            0.0,  # 8 column slalom
            0.0,  # 9
            0.25, # 10
            0.45, # 11
            0.45, # 12
            0.0,  # 13
        ]
        # dynamic obstacle stages (2 wall-push, 3 corner-trap) are very hard to complete -> they use survival success.
        self._survival_stages = {2, 3}

    @property
    def stage_config_name(self) -> str:
        return self._stage_names[self.current_stage]

    @property
    def chaser_speed(self) -> float:
        return self._chaser_speeds[self.current_stage]

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps[self.current_stage]

    @property
    def is_survival_stage(self) -> bool:
        return self.current_stage in self._survival_stages

    def configure_engine(self, engine, goal: np.ndarray) -> dict:
        """
        Sets up obstacles for the current stage with domain randomization.

        Parameters:
        - engine : SuperharmonicFieldEngine
        - goal : np.ndarray - goal position
        Returns:
        - meta : dict with start_pos, chaser_idx, ws_lo, ws_hi
        """
        engine.obstacles.clear()
        stage = self.current_stage
        jitter = lambda: np.random.randn(3) * 0.25
        ws_lo, ws_hi = -3.0, 3.0
        chaser_idx = -1

        if stage == 0:
            # BOOTSTRAP: no obstacles, random start, just navigate to goal
            start = np.random.uniform(-2.0, 0.5, size=3)
            start = np.clip(start, ws_lo + 0.3, ws_hi - 0.3)

        elif stage == 1:
            # STATIC BLOCKER : 1 static obstacle on the path to the goal
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            t_blocker = np.random.uniform(0.35, 0.65)
            line_start = np.array([-2.0, -2.0, 0.5])
            blocker_pos = line_start + t_blocker * (goal - line_start)
            blocker_pos += np.random.randn(3) * 0.25
            blocker_pos = np.clip(blocker_pos, ws_lo + 0.4, ws_hi - 0.4)
            blocker_r = np.random.uniform(0.40, 0.55)
            engine.add_obstacle(blocker_pos, blocker_r, velocity=None)

        elif stage == 2:
            # WALL PUSH : 1 obstacle pushes drone toward a wall
            axis = np.random.randint(3)
            sign = np.random.choice([-1, 1])

            if sign > 0:
                ws_hi = 2.0 + np.random.uniform(0.0, 1.0)
            else:
                ws_lo = -2.0 - np.random.uniform(0.0, 1.0)

            start = np.random.uniform(-1.0, 1.0, size=3)
            wall_pos = ws_hi if sign > 0 else ws_lo
            start[axis] = wall_pos - sign * 0.8
            start = np.clip(start, ws_lo + 0.2, ws_hi - 0.2)

            obs_pos = start.copy()
            obs_pos[axis] -= sign * 0.9
            obs_vel = np.zeros(3)
            obs_vel[axis] = sign * 0.20
            engine.add_obstacle(obs_pos + jitter(), 0.2, velocity=obs_vel)

        elif stage == 3:
            # CORNER TRAP: 2 obstacles pin drone into a corner
            corner_signs = np.array([np.random.choice([-1, 1]) for _ in range(3)])

            ws_lo = -2.5 - np.random.uniform(0.0, 0.5)
            ws_hi = 2.5 + np.random.uniform(0.0, 0.5)
            corner = corner_signs * np.array([abs(ws_hi) - 0.5, abs(ws_hi) - 0.5, abs(ws_hi) - 0.5])

            start = corner * 0.85 + jitter() * 0.2
            start = np.clip(start, ws_lo + 0.3, ws_hi - 0.3)

            for i in range(2):
                obs_pos = start.copy()
                ax = i % 3
                obs_pos[ax] -= corner_signs[ax] * 0.9
                obs_vel = np.zeros(3)
                obs_vel[ax] = corner_signs[ax] * 0.25
                engine.add_obstacle(obs_pos + jitter(), 0.38, velocity=obs_vel)

        elif stage == 4:
            # STATIC U-TRAP - 3-sphere U pointing back toward the start
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            theta = np.random.uniform(0, 2 * np.pi)
            phi_tilt = np.random.uniform(-np.pi / 6, np.pi / 6)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cos_p = np.cos(phi_tilt)
            mid_pt = np.array([0.2, 0.2, 1.5]) + np.random.randn(3) * 0.20
            ce = np.array([0.4 * cos_t * cos_p, 0.4 * sin_t * cos_p, 0.4 * np.sin(phi_tilt)])
            engine.add_obstacle(mid_pt + ce, 0.35, velocity=None)
            arm1 = np.array([-0.7 * sin_t, 0.7 * cos_t, 0.0])
            engine.add_obstacle(mid_pt + ce + arm1, 0.30, velocity=None)
            engine.add_obstacle(mid_pt + ce - arm1, 0.30, velocity=None)

        elif stage == 5:
            # 2 CLOSE MOVERS
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.5
            engine.add_obstacle(np.array([-0.3, 0.0, 0.8]) + jitter(), 0.40, velocity=np.array([0.30, 0.0, 0.0]))
            engine.add_obstacle(np.array([0.3, 0.3, 0.0]) + jitter(), 0.35, velocity=np.array([0.0, 0.0, 0.30]))

        elif stage == 6:
            # 4 MIXED MOVERS 
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.5
            engine.add_obstacle(np.array([-0.5, 0.0, 1.0]) + jitter(), 0.38, velocity=np.array([0.30, 0.0, 0.0]))
            engine.add_obstacle(np.array([0.5, 0.5, -0.3]) + jitter(), 0.35, velocity=np.array([0.0, 0.0, 0.30]))
            engine.add_obstacle(np.array([0.0, 1.0, 0.5]) + jitter(), 0.35, velocity=np.array([0.0, -0.25, 0.0]))
            engine.add_obstacle(np.array([1.0, -0.5, 1.5]) + jitter(), 0.30, velocity=np.array([-0.20, 0.20, 0.0]))

        elif stage == 7:
            # STATIC SADDLE / CLUSTER 4 static obstacles arranged in a saddle geometry
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            mid_pt = np.array([-0.25, -0.25, 1.5]) + np.random.randn(3) * 0.20
            # cluster A (one side of the start->goal line)
            engine.add_obstacle(mid_pt + np.array([-0.1, 0.9, 0.0]) + jitter() * 0.3, 0.30, velocity=None)
            engine.add_obstacle(mid_pt + np.array([0.3, 1.1, 0.1]) + jitter() * 0.3, 0.25, velocity=None)
            # cluster B (other side)
            engine.add_obstacle(mid_pt + np.array([0.9, -0.1, 0.0]) + jitter() * 0.3, 0.30, velocity=None)
            engine.add_obstacle(mid_pt + np.array([1.1, 0.3, -0.1]) + jitter() * 0.3, 0.25, velocity=None)

        elif stage == 8:
            # COLUMN SLALOM: 1-2 floor-to-ceiling rectangular columns
            # roughly on the start->goal line. Static; no chaser. Forces the
            # agent to navigate around a box obstacle in the xy plane.
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            line_start = np.array([-2.0, -2.0, 0.5])
            n_cols = np.random.choice([1, 2])
            for i in range(n_cols):
                t_col = np.random.uniform(0.30 + 0.20 * i, 0.55 + 0.20 * i)
                center_3d = line_start + t_col * (goal - line_start)
                lateral = np.array([np.random.choice([-1.0, 1.0]) * np.random.uniform(0.0, 0.25), np.random.choice([-1.0, 1.0]) * np.random.uniform(0.0, 0.25)])
                center_xy = np.array([center_3d[0], center_3d[1]]) + lateral
                center_xy = np.clip(center_xy, ws_lo + 0.5, ws_hi - 0.5)
                hx = np.random.uniform(0.25, 0.45)
                hy = np.random.uniform(0.25, 0.45)
                col = RectangularColumn(center_xy=center_xy, half_extents_xy=np.array([hx, hy], dtype=float),z_lo=ws_lo, z_hi=ws_hi, velocity_xy=None, label=f'col{i}')
                engine.obstacles.append(col)

        elif stage == 9:
            # 6 DENSE MOVERS 
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            configs = [
                ([-0.5, 0.0, 1.0], 0.38, [0.30, 0.0, 0.0]),
                ([0.5, 0.5, -0.3], 0.35, [0.0, 0.0, 0.30]),
                ([0.0, 1.0, 0.5], 0.35, [-0.25, 0.0, 0.0]),
                ([1.0, 0.0, 1.5], 0.30, [0.0, -0.25, 0.0]),
                ([-1.0, -1.0, 1.0], 0.33, [0.20, 0.20, 0.0]),
                ([0.5, -0.5, 0.0], 0.30, [0.0, 0.20, 0.20]),
            ]
            for pos, r, vel in configs:
                speed_jitter = 1.0 + np.random.uniform(-0.2, 0.2)
                engine.add_obstacle(np.array(pos) + jitter(), r, velocity=np.array(vel) * speed_jitter)

        elif stage == 10:
            # 3 MOVERS + SLOW CHASER 
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.5
            engine.add_obstacle(np.array([-0.5, 0.0, 1.0]) + jitter(), 0.40, velocity=np.array([0.30, 0.0, 0.0]))
            engine.add_obstacle(np.array([0.5, 0.5, -0.5]) + jitter(), 0.35, velocity=np.array([0.0, 0.0, 0.30]))
            engine.add_obstacle(np.array([0.8, 1.8, 1.8]) + jitter(), 0.30, velocity=None)
            engine.add_obstacle(np.array([0.0, 0.0, 1.2]) + jitter(), 0.35, velocity=np.zeros(3))
            chaser_idx = 3

        elif stage == 11:
            # 5 MOVERS + FAST CHASER 
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.5
            engine.add_obstacle(np.array([-0.5, 0.0, 1.0]) + jitter(), 0.38, velocity=np.array([0.35, 0.0, 0.0]))
            engine.add_obstacle(np.array([0.5, 0.5, -0.5]) + jitter(), 0.35, velocity=np.array([0.0, 0.0, 0.35]))
            engine.add_obstacle(np.array([0.0, 1.0, 0.5]) + jitter(), 0.33, velocity=np.array([-0.25, 0.0, 0.0]))
            engine.add_obstacle(np.array([1.0, -0.5, 1.5]) + jitter(), 0.30, velocity=np.array([0.0, 0.25, 0.0]))
            engine.add_obstacle(np.array([-1.0, -1.0, 1.0]) + jitter(), 0.33, velocity=np.array([0.20, 0.20, 0.0]))
            engine.add_obstacle(np.array([0.0, 0.0, 1.2]) + jitter(), 0.35, velocity=np.zeros(3))
            chaser_idx = 5

        elif stage == 12:
            # 6 MOVERS + FAST CHASER
            start = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.4
            configs = [
                ([-0.5, 0.0, 1.0], 0.38, [0.35, 0.0, 0.0]),
                ([0.5, 0.5, -0.5], 0.35, [0.0, 0.0, 0.35]),
                ([0.0, 1.0, 0.5], 0.33, [-0.30, 0.0, 0.0]),
                ([1.0, -0.5, 1.5], 0.30, [0.0, -0.25, 0.0]),
                ([-1.0, -1.0, 1.0], 0.33, [0.25, 0.25, 0.0]),
                ([0.5, -0.5, 0.0], 0.30, [0.0, 0.20, 0.25]),
            ]
            for pos, r, vel in configs:
                speed_jitter = 1.0 + np.random.uniform(-0.2, 0.2)
                engine.add_obstacle(np.array(pos) + jitter(), r, velocity=np.array(vel) * speed_jitter)
            engine.add_obstacle(np.array([0.0, 0.0, 1.2]) + jitter(), 0.35, velocity=np.zeros(3))
            chaser_idx = 6

        elif stage == 13:
            # FINAL APPROACH: close start, no obstacles -
            # forces last-metre precision
            offset = np.random.uniform(0.3, 1.0)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction) + 1e-8
            start = goal + direction * offset
            start = np.clip(start, ws_lo + 0.3, ws_hi - 0.3)

        else:
            start = np.array([-2.0, -2.0, 0.5])

        if stage not in {2, 3}:
            start = np.clip(start, ws_lo + 0.3, ws_hi - 0.3)

        return {
            'start_pos': start,
            'chaser_idx': chaser_idx,
            'ws_lo': ws_lo,
            'ws_hi': ws_hi,
        }

    def record_episode(self, goal_reached: bool, collision: bool,final_d_goal: float, survived: bool) -> bool:
        """
        Records episode outcome and checks for stage promotion.

        Parameters:
        - goal_reached : bool
        - collision : bool
        - final_d_goal : float
        - survived : bool - reached max steps without collision
        Returns:
        - promoted : bool - whether the agent advanced to the next stage
        """
        if self.is_survival_stage:
            #survival stages: success = no collision (regardless of goal)
            success = not collision
        else:
            #navigation stages: success = goal OR close + no collision
            success = goal_reached or (not collision and final_d_goal < 2.5)

        self.episode_results.append(success)
        self.episodes_in_stage += 1

        # lower threshold for early stages
        threshold = (self.early_promotion_threshold
                     if self.current_stage in self._survival_stages
                     else self.promotion_threshold)

        if (self.episodes_in_stage >= self.min_episodes and self.current_stage < self.N_STAGES - 1):
            recent = self.episode_results[-self.window_size:]
            rate = sum(recent) / len(recent)
            if rate >= threshold:
                self.current_stage += 1
                self.episodes_in_stage = 0
                self.stage_history.append({
                    'from_stage': self.current_stage - 1,
                    'to_stage': self.current_stage,
                    'episodes': len(self.episode_results),
                    'success_rate': rate,
                })
                return True
        return False


#escape environment with all reward fixes

class EscapeEnvironment:
    """
    Observation layout (13 dims, field-only):
        [0:3]   relative goal direction (unit-ish, via current distance)
        [3:6]   raw gradient ∇Φ (magnitude encodes urgency)
        [6]     potential Φ (tanh-compressed)
        [7]     ∂Φ/∂t (tanh-compressed)
        [8]     time fraction
        [9:12]  wall distances per axis (tanh-compressed)
        [12]    goal proximity (tanh-compressed absolute distance)

    Parameters:
    - field_engine : SuperharmonicFieldEngine
    - reward_shaper : PBRSRewardShaper
    - max_escape_speed : float
    - max_episode_steps : int
    - workspace_bounds : tuple
    - dt : float
    - chaser_idx : int
    - chaser_speed : float
    - progress_weight : float - weight for goal-progress reward
    """

    def __init__(self, field_engine, reward_shaper: PBRSRewardShaper, max_escape_speed: float = 1.5, max_episode_steps: int = 100, workspace_bounds: tuple = (-3.0, 3.0),
                 dt: float = 0.1, chaser_idx: int = -1, chaser_speed: float = 0.0, progress_weight: float = 1.5):
        self.engine = field_engine
        self.shaper = reward_shaper
        self.max_escape_speed = max_escape_speed
        self.max_episode_steps = max_episode_steps
        self.ws_lo = workspace_bounds[0]
        self.ws_hi = workspace_bounds[1]
        self.dt = dt
        self.chaser_idx = chaser_idx
        self.chaser_speed = chaser_speed
        self.progress_weight = progress_weight

        # obs: 3(goal) + 3(grad_raw) + 1(Phi) + 1(dPhi/dt) + 1(time) + 3(walls) + 1(d_goal)
        self.obs_dim = 13
        self.act_dim = 3
        self.agent_pos = None
        self.step_count = 0
        self.initial_d_goal = None
        self._prev_d_goal = None
        self.wall_margin = 0.15

    def _wall_distances(self, pos: np.ndarray) -> np.ndarray:
        d_lo = pos - self.ws_lo
        d_hi = self.ws_hi - pos
        return np.minimum(d_lo, d_hi)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float64)
        pos = self.agent_pos

        # goal direction - normalized by current distance (not initial_d_goal);
        # signal stays strong and directional right up to the goal
        rel_goal = self.engine.goal_position - pos
        d_goal_now = np.linalg.norm(rel_goal)
        obs[0:3] = rel_goal / max(d_goal_now, 0.5)

        # raw gradient - magnitude encodes urgency
        grad = self.engine.compute_gradient(pos)
        obs[3:6] = grad

        # potential Phi (tanh-compressed)
        phi = self.engine.compute_potential(pos)
        obs[6] = np.tanh(phi / 5.0)

        # dPhi/dt (tanh-compressed)
        dphi_dt = self.engine.compute_dphi_dt(pos)
        obs[7] = np.tanh(dphi_dt)

        # time fraction
        obs[8] = min(self.step_count / self.max_episode_steps, 1.0)

        # wall distances (tanh-compressed per axis)
        wall_d = self._wall_distances(pos)
        ws_half = (self.ws_hi - self.ws_lo) / 2.0
        obs[9:12] = np.tanh(wall_d / max(ws_half * 0.3, 0.1))

        # goal proximity: tanh-compressed absolute distance (dim 12)
        # lets the agent sense closeness independently of direction
        obs[12] = np.tanh(d_goal_now / 3.0)

        return obs

    def reset(self, agent_pos: np.ndarray = None) -> np.ndarray:
        if agent_pos is not None:
            self.agent_pos = np.array(agent_pos, dtype=float)
        else:
            noise = np.random.randn(3) * 0.5
            self.agent_pos = np.array([-1.5, -1.5, 0.5]) + noise
            self.agent_pos = np.clip(self.agent_pos, self.ws_lo + 0.3, self.ws_hi - 0.3)

        self.step_count = 0
        self.initial_d_goal = max(np.linalg.norm(self.agent_pos - self.engine.goal_position), 1.0)
        self._prev_d_goal = self.initial_d_goal
        self.shaper.reset(self.agent_pos)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        velocity = action * self.max_escape_speed
        self.agent_pos = self.agent_pos + velocity * self.dt
        self.agent_pos = np.clip(self.agent_pos, self.ws_lo, self.ws_hi)
        self.step_count += 1

        # chaser pursuit
        ci = self.chaser_idx
        if 0 <= ci < len(self.engine.obstacles) and self.chaser_speed > 0:
            chaser = self.engine.obstacles[ci]
            if not isinstance(chaser, RectangularColumn):
                chase_vec = self.agent_pos - chaser.position
                cd = np.linalg.norm(chase_vec)
                if cd > 1e-6:
                    chaser.velocity = (chase_vec / cd) * self.chaser_speed

        # distances
        d_goal = np.linalg.norm(self.agent_pos - self.engine.goal_position)
        collision = False
        min_clearance = float('inf')
        for o in self.engine.obstacles:
            d = _obstacle_surface_distance(o, self.agent_pos)
            min_clearance = min(min_clearance, d)
            if d < 0.05:
                collision = True
                break

        wall_d = self._wall_distances(self.agent_pos)
        min_wall = wall_d.min()
        wall_collision = min_wall < 0.02

        goal_reached = d_goal < 0.35

        #reward decomposition

        # (a) survival: penalties only
        r_survival = 0.0
        if collision:
            r_survival -= 50.0
        if wall_collision:
            r_survival -= 30.0
        if min_wall < self.wall_margin:
            r_survival -= 3.0 * (self.wall_margin - min_wall) / self.wall_margin

        # (b) navigation: step cost + goal bonus + progress
        r_navigation = -0.1
        if goal_reached:
            r_navigation += 100.0

        # dense progress reward: bounded, stable signal
        r_progress = self.progress_weight * (self._prev_d_goal - d_goal) / self.initial_d_goal
        self._prev_d_goal = d_goal
        r_navigation += r_progress

        # proximity bonus: dense signal that grows as agent closes in on the goal
        # compensates for the near-zero gradient and progress signal in the last metre
        r_navigation += 2.0 * max(0.0, 1.0 - d_goal / 1.0)

        # (c) task reward = survival + navigation
        r_task = r_survival + r_navigation

        done = collision or wall_collision or goal_reached or \
               self.step_count >= self.max_episode_steps

        # (d) PBRS shaping (already clipped in PBRSRewardShaper)
        r_shaped, r_task_out, f_shaping = self.shaper.compute_shaped_reward(
            self.agent_pos, r_task, done)

        # (e) dPhi/dt anticipation bonus (kept, but scaled down)
        dphi_dt = self.engine.compute_dphi_dt(self.agent_pos)
        if dphi_dt > 0 and not done:
            grad = self.engine.compute_gradient(self.agent_pos)
            gn = np.linalg.norm(grad)
            vn = np.linalg.norm(velocity)
            if gn > 1e-8 and vn > 1e-8:
                align = -np.dot(velocity, grad) / (vn * gn)
                r_shaped += 0.3 * dphi_dt * max(align, 0.0)  # reduced from 0.5

        for o in self.engine.obstacles:
            if isinstance(o, RectangularColumn):
                o.update_position(self.dt, self.ws_lo, self.ws_hi)
            else:
                o.update_position(self.dt)

        # bounce non-chaser obstacles (skip columns, they bounce themselves)
        for oi, o in enumerate(self.engine.obstacles):
            if isinstance(o, RectangularColumn):
                continue
            if oi == ci:
                for dim in range(3):
                    o.position[dim] = np.clip(o.position[dim], self.ws_lo + o.radius, self.ws_hi - o.radius)
                continue
            if o.velocity is not None:
                for dim in range(3):
                    lo = self.ws_lo + o.radius
                    hi = self.ws_hi - o.radius
                    if o.position[dim] <= lo or o.position[dim] >= hi:
                        o.velocity[dim] *= -1
                        o.position[dim] = np.clip(o.position[dim], lo, hi)

        survived = (self.step_count >= self.max_episode_steps and not collision and not wall_collision)

        info = {
            'd_goal': d_goal,
            'min_clearance': min_clearance,
            'min_wall': min_wall,
            'collision': collision or wall_collision,
            'goal_reached': goal_reached,
            'survived': survived,
            'r_task': r_task_out,
            'r_progress': r_progress,
            'f_shaping': f_shaping,
            'r_shaped': r_shaped,
        }
        return self._get_obs(), r_shaped, done, info


# engine factory

def make_engine(goal: np.ndarray) -> SuperharmonicFieldEngine:
    return SuperharmonicFieldEngine(
        goal_position=goal, a_att=0.1, a_rep=1.0,
        n_power=2.0, danger_distance=1.0,
        workspace_lo=-3.0, workspace_hi=3.0,
        a_wall=0.5, wall_power=2.0, wall_danger=0.8,
    )


# training loop 

def train(n_episodes: int = 20000, save_dir: str = "ppo_escape_v9_checkpoints", use_curriculum: bool = True, gamma: float = 0.99, potential_scale: float = 1.0, 
          pbrs_clip: float = 5.0, progress_weight: float = 1.5, verbose: bool = True):

    goal = np.array([2.0, 2.0, 2.0])
    engine = make_engine(goal)
    shaper = PBRSRewardShaper(engine, gamma=gamma, potential_scale=potential_scale, clip_bound=pbrs_clip)

    curriculum = None
    if use_curriculum:
        curriculum = CurriculumManager(
            promotion_threshold=0.65,
            early_promotion_threshold=0.55,
            window_size=50,
            min_episodes_per_stage=1000,
        )

    env = EscapeEnvironment(
        field_engine=engine,
        reward_shaper=shaper,
        max_escape_speed=1.5,
        max_episode_steps=200,
        chaser_idx=-1,
        chaser_speed=0.0,
        progress_weight=progress_weight,
    )

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=3,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=gamma,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.02,
        n_epochs=10,
        batch_size=64,
        hidden_sizes=[64, 64],
    )

    # running return normaliser
    return_normaliser = RunningMeanStd()

    steps_per_update = 2048

    if verbose:
        print("=" * 65)
        print("PPO ESCAPE TRAINING v9 - Column-aware curriculum + collision")
        print("=" * 65)
        print(f"  obs_dim          : {env.obs_dim}")
        print(f"  hidden           : [64, 64]")
        print(f"  obs_features     : goal(3) + grad(3) + Phi + dPhi/dt + time + walls(3)")
        print(f"  episodes         : {n_episodes}")
        print(f"  potential_scale  : {potential_scale}  (was 10.0 in v5)")
        print(f"  pbrs_clip        : {pbrs_clip}")
        print(f"  progress_weight  : {progress_weight}")
        if curriculum:
            for i, name in enumerate(curriculum._stage_names):
                print(f"    {name}  (steps={curriculum._max_steps[i]})")
        print()

    episode_rewards = []
    episode_goals = 0
    episode_collisions = 0
    total_steps = 0
    goal_reached_results = []  # strict goal-reach outcomes for display only

    for episode in range(n_episodes):
        if curriculum:
            meta = curriculum.configure_engine(engine, goal)
            start_pos = meta['start_pos']
            env.chaser_idx = meta['chaser_idx']
            env.chaser_speed = curriculum.chaser_speed
            env.max_episode_steps = curriculum.max_episode_steps
            env.ws_lo = meta['ws_lo']
            env.ws_hi = meta['ws_hi']
            engine.set_workspace(meta['ws_lo'], meta['ws_hi'])
        else:
            start_pos = np.array([-2.0, -2.0, 0.5]) + np.random.randn(3) * 0.5
            start_pos = np.clip(start_pos, -2.5, 2.5)

        for o in engine.obstacles:
            if isinstance(o, RectangularColumn):
                continue
            o.position += np.random.randn(3) * 0.15

        obs = env.reset(agent_pos=start_pos)
        ep_reward = 0.0

        for step in range(env.max_episode_steps):
            action, log_prob, value, pre_tanh = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.buffer.add(obs, action, pre_tanh, reward, value, log_prob, done)

            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if (total_steps % steps_per_update == 0 and len(agent.buffer.observations) >= agent.batch_size):
                _, _, last_val, _ = agent.select_action(obs)
                agent.buffer.compute_returns_and_advantages(agent, last_val)

                # normalise returns for stable value learning
                if agent.buffer.returns is not None:
                    return_normaliser.update(agent.buffer.returns)
                    agent.buffer.returns = return_normaliser.normalize(agent.buffer.returns)

                agent.update()

            if done:
                if info["collision"]:
                    episode_collisions += 1
                if info["goal_reached"]:
                    episode_goals += 1
                break

        goal_reached_results.append(bool(info.get("goal_reached", False)))
        episode_rewards.append(ep_reward)

        promoted = False
        if curriculum:
            promoted = curriculum.record_episode(
                info.get("goal_reached", False),
                info.get("collision", False),
                info.get("d_goal", 99.0),
                info.get("survived", False),
            )

        if verbose and (episode % 100 == 0 or promoted):
            recent_r = (np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards))
            stage_name = curriculum.stage_config_name if curriculum else "Full"

            # rolling goal-reach rate (strict: only episodes where goal was reached)
            recent_goals = goal_reached_results[-50:]
            success_rate = sum(recent_goals) / max(len(recent_goals), 1)

            print(f"  Ep {episode:5d}  r={ep_reward:+8.1f}  avg={recent_r:+8.1f}  "
                  f"goals={episode_goals}  coll={episode_collisions}  "
                  f"sr={success_rate:.2f}  stage={stage_name}")
            if promoted:
                print(f"    >>> PROMOTED to {curriculum.stage_config_name}!")

    if len(agent.buffer.observations) >= agent.batch_size:
        _, _, last_val, _ = agent.select_action(obs)
        agent.buffer.compute_returns_and_advantages(agent, last_val)
        if agent.buffer.returns is not None:
            return_normaliser.update(agent.buffer.returns)
            agent.buffer.returns = return_normaliser.normalize(agent.buffer.returns)
        agent.update()

    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)

    if verbose:
        print(f"\n  Done. {n_episodes} ep, {total_steps} steps.")
        print(f"  Goals: {episode_goals}/{n_episodes} ({100*episode_goals/max(n_episodes,1):.1f}%)")
        print(f"  Collisions: {episode_collisions}/{n_episodes} ({100*episode_collisions/max(n_episodes,1):.1f}%)")
        if curriculum:
            print(f"  Final stage: {curriculum.stage_config_name}")
            for sh in curriculum.stage_history:
                print(f"    {sh['from_stage']}->{sh['to_stage']} at ep {sh['episodes']} (rate={sh['success_rate']:.2f})")
        print(f"  Saved to {save_dir}/")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO escape v9 (column-aware)")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--save-dir", type=str, default="ppo_escape_v9_10k")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--potential-scale", type=float, default=1.0)
    parser.add_argument("--pbrs-clip", type=float, default=5.0)
    parser.add_argument("--progress-weight", type=float, default=1.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        save_dir=args.save_dir,
        use_curriculum=not args.no_curriculum,
        potential_scale=args.potential_scale,
        pbrs_clip=args.pbrs_clip,
        progress_weight=args.progress_weight,
        verbose=not args.quiet,
    )