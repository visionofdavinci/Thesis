"""
Navigation Controller for Hybrid Potential Field + PPO Switching

Components:
1. StuckDetector: sliding-window detection of freeze states and limit cycles
2. ModeController: three-mode Schmitt trigger with PPO override
3. build_escape_observation(): observation builder (obs_dim=12)
   matching train_ppo_escape.py EscapeEnvironment._get_obs()

Observation layout (v6, field-only):
    [0:3]   relative goal position (normalized)
    [3:6]   raw gradient ∇Φ (magnitude encodes urgency)
    [6]     potential Φ (tanh-compressed)
    [7]     ∂Φ/∂t (tanh-compressed)
    [8]     time fraction
    [9:12]  wall distances per axis (tanh-compressed)
"""

import numpy as np
from typing import Optional


# stuck detection

class StuckDetector:
    """
    Detects freeze, limit cycle, and stagnation failure modes.

    Parameters:
    - window_size : int - sliding window length
    - progress_threshold : float - min net goal-progress over window
    - cycle_ratio : float - path_length/displacement ratio for limit cycle
    - stagnation_window : int - steps without improving best d_goal
    - stagnation_threshold : float - min improvement to reset stagnation
    """

    def __init__(self, window_size: int = 30, progress_threshold: float = 0.15,
                 cycle_ratio: float = 5.0, stagnation_window: int = 120,
                 stagnation_threshold: float = 0.25):
        self.window_size = window_size
        self.progress_threshold = progress_threshold
        self.cycle_ratio = cycle_ratio
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold

        self.goal_distances = []
        self.positions = []
        self.stuck_counter = 0
        self.is_stuck = False
        self.stuck_type = 'none'
        self._best_d_goal = float('inf')
        self._steps_since_best = 0

    def update(self, pos: np.ndarray, d_goal: float) -> bool:
        self.positions.append(pos.copy())
        self.goal_distances.append(d_goal)

        if d_goal < self._best_d_goal - self.stagnation_threshold:
            self._best_d_goal = d_goal
            self._steps_since_best = 0
        else:
            self._steps_since_best += 1

        if self._steps_since_best >= self.stagnation_window:
            self.stuck_counter += 1
            self.is_stuck = True
            self.stuck_type = 'stagnation'
            return True

        if len(self.goal_distances) < self.window_size:
            self.is_stuck = False
            self.stuck_type = 'none'
            return False

        d_old = self.goal_distances[-self.window_size]
        d_now = self.goal_distances[-1]
        net_progress = d_old - d_now

        recent_positions = self.positions[-self.window_size:]
        path_length = sum(
            np.linalg.norm(recent_positions[i+1] - recent_positions[i])
            for i in range(len(recent_positions) - 1)
        )
        net_displacement = np.linalg.norm(recent_positions[-1] - recent_positions[0])

        if path_length < 0.05:
            self.stuck_counter += 1; self.is_stuck = True; self.stuck_type = 'freeze'
            return True

        ratio = path_length / max(net_displacement, 1e-6)
        if net_progress < self.progress_threshold and ratio > self.cycle_ratio:
            self.stuck_counter += 1; self.is_stuck = True; self.stuck_type = 'limit_cycle'
            return True

        if net_progress > self.progress_threshold:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        self.is_stuck = False; self.stuck_type = 'none'
        return self.is_stuck

    def reset(self):
        self.goal_distances.clear(); self.positions.clear()
        self.stuck_counter = 0; self.is_stuck = False; self.stuck_type = 'none'
        self._best_d_goal = float('inf'); self._steps_since_best = 0


# mode controller

class ModeController:
    """
    Three-mode Schmitt trigger: subharmonic, superharmonic, ppo.

    Parameters:
    - high_threshold : float - dPhi/dt above this triggers superharmonic
    - low_threshold : float - dPhi/dt below this returns to subharmonic
    - min_dwell : int - minimum steps before switching
    """

    def __init__(self, high_threshold: float = 0.2, low_threshold: float = 0.05,
                 min_dwell: int = 10):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_dwell = min_dwell
        self.mode = 'subharmonic'
        self.steps_in_mode = 0
        self.switch_count = 0
        self.mode_history = []

    def update(self, dphi_dt: float, stuck: bool) -> str:
        self.steps_in_mode += 1

        if stuck:
            if self.mode != 'ppo':
                self.mode = 'ppo'; self.steps_in_mode = 0; self.switch_count += 1
        elif self.mode == 'ppo':
            self.mode = 'superharmonic' if dphi_dt > self.high_threshold else 'subharmonic'
            self.steps_in_mode = 0; self.switch_count += 1
        elif self.steps_in_mode >= self.min_dwell:
            if self.mode == 'subharmonic' and dphi_dt > self.high_threshold:
                self.mode = 'superharmonic'; self.steps_in_mode = 0; self.switch_count += 1
            elif self.mode == 'superharmonic' and dphi_dt < self.low_threshold:
                self.mode = 'subharmonic'; self.steps_in_mode = 0; self.switch_count += 1

        self.mode_history.append(self.mode)
        return self.mode

    def reset(self):
        self.mode = 'subharmonic'; self.steps_in_mode = 0
        self.switch_count = 0; self.mode_history.clear()


# observation builder (obs_dim=12, field-only)

def build_escape_observation(agent_pos: np.ndarray, field_engine,
                              step_count: int,
                              max_episode_steps: int = 150,
                              initial_d_goal: float = 5.85,
                              ws_lo: float = -3.0,
                              ws_hi: float = 3.0,
                              **kwargs) -> np.ndarray:
    """
    Builds 12-dim field-only observation matching the training f().
    No raw obstacle data -> the potential field is the abstraction.

    Parameters:
    - agent_pos : np.ndarray - current drone position
    - field_engine : engine with compute_potential/gradient/dphi_dt
    - step_count : int - current step
    - max_episode_steps : int - max steps (for time normalization)
    - initial_d_goal : float - initial distance to goal
    - ws_lo : float - workspace lower bound
    - ws_hi : float - workspace upper bound
    - **kwargs : ignored (for backward compatibility)
    Returns:
    - obs : np.ndarray of shape (12,)
    """
    obs = np.zeros(12, dtype=np.float64)
    pos = np.asarray(agent_pos, dtype=float)

    #relative goal (normalized)
    rel_goal = field_engine.goal_position - pos
    obs[0:3] = rel_goal / max(initial_d_goal, 1.0)

    #raw gradient — magnitude encodes urgency
    grad = field_engine.compute_gradient(pos)
    obs[3:6] = grad

    #potential Phi (tanh-compressed)
    phi = field_engine.compute_potential(pos)
    obs[6] = np.tanh(phi / 5.0)

    #dPhi/dt (tanh-compressed)
    dphi_dt = field_engine.compute_dphi_dt(pos)
    obs[7] = np.tanh(dphi_dt)

    #time fraction
    obs[8] = min(step_count / max_episode_steps, 1.0)

    #wall distances per axis (tanh-compressed)
    d_lo = pos - ws_lo
    d_hi = ws_hi - pos
    wall_d = np.minimum(d_lo, d_hi)
    ws_half = (ws_hi - ws_lo) / 2.0
    obs[9:12] = np.tanh(wall_d / max(ws_half * 0.3, 0.1))

    return obs