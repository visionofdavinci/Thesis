"""
Pure PPO Navigation Training - End-to-End Goal-Reaching Baseline (v6)
"""

import argparse
import os
import time
import json
import numpy as np
from typing import List, Tuple, Optional, Dict



from engines.navigation_controller import build_escape_observation
from engines.superharmonic_field_engine import SuperharmonicFieldEngine
from engines.rectangular_column import RectangularColumn



def _obstacle_surface_distance(obstacle, agent_pos: np.ndarray) -> float:
    """
    Distance from agent to the nearest surface point of an obstacle.
    Routes RectangularColumn (and any other obstacle with surface_distance), falls back to the spherical formula otherwise.
    """
    if hasattr(obstacle, 'surface_distance'):
        return float(obstacle.surface_distance(agent_pos))
    return float(np.linalg.norm(agent_pos - obstacle.position) - obstacle.radius)


# hyperparameters
HYPERPARAMS = {
    # PPO core
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "clip_eps_vf": 0.2,
    "vf_coef": 0.5,
    "entropy_coef_init": 0.01,
    "entropy_coef_final": 0.001,
    "max_grad_norm": 0.5,
    "lr_init": 3e-4,
    "lr_final": 1e-5,
    "n_epochs": 10,
    "batch_size": 128,
    "rollout_steps": 2048,
    "target_kl": 0.03,

    # network
    "hidden_sizes": [128, 128],
    "obs_dim": 13,
    "act_dim": 3,

    # environment
    "dt": 0.1,
    "max_velocity": 1.5,
    "workspace_size_min": 2.5,
    "workspace_size_max": 4.0,
    "goal_tolerance": 0.35,
    "collision_threshold": 0.05,
    "wall_collision_threshold": 0.02,
    "min_start_goal_distance": 3.5,

    "r_goal":               1000.0,    # was 10.0
    "r_collision":         -50.0,    # was -10.0
    "r_timeout":           -10.0,    # was -2.0
    "r_step":               -0.2,
    "r_progress_scale":      1.0,    # was 5.0, now applied to NORMALISED progress
    "r_smooth_weight":       0.05,

    # curriculum thresholds 
    "curriculum_stage1_start": 2500,
    "curriculum_stage2_start": 7500,

    # column probabilities 
    "column_prob_stage1": 0.20,
    "column_prob_stage2": 0.40,

    # normalisation toggles
    "normalize_obs":     True,
    "normalize_returns": True,

    # training
    "total_episodes": 25000,
    "log_interval": 50,
    "save_interval": 500,
    "eval_interval": 250,
    "eval_episodes": 20,
    "seed": 0,
}

#running mean / variance (Welford, parallel-safe)
class RunningMeanStd:
    """
    Maintains a running mean and variance over a stream of samples.

    Used for both observation normalisation (shape=(obs_dim,)) and return normalisation (shape=()). Initialised with var=1, count=1e-4
    so the first few normalised samples are not divided by ~0.
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1)
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0)
        b_count = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    def state_dict(self) -> Dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ReturnNormalizer:
    """
    Reward normalisation by the running std of discounted returns.

    Following Engstrom et al. 2020: maintain a per-rollout discounted return tracker, update an RMS over those values, and divide each
    reward by sqrt(var). Resets the discounted-return tracker on episode termination.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.ret_rms = RunningMeanStd(())
        self.discounted_return = 0.0

    def normalize(self, reward: float, done: bool) -> float:
        self.discounted_return = self.gamma * self.discounted_return + reward
        self.ret_rms.update(np.array([self.discounted_return]))
        std = float(np.sqrt(self.ret_rms.var + 1e-8))
        norm_reward = reward / std
        if done:
            self.discounted_return = 0.0
        return float(norm_reward)


# the networks
class PPOActorCritic:
    """
    Separate actor-critic with ReLU hidden layers.
    Actor outputs tanh-squashed mean + learned log_std.
    Critic outputs scalar V(s).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int], seed: int = 0, normalize_obs: bool = True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.normalize_obs = normalize_obs
        self.rng = np.random.RandomState(seed)

        self.actor_weights, self.actor_biases = self._init_network([obs_dim] + hidden_sizes + [act_dim])
        self.log_std = np.zeros(act_dim) - 0.5

        self.critic_weights, self.critic_biases = self._init_network([obs_dim] + hidden_sizes + [1])

        #running stats for obs normalisation
        self.obs_rms = RunningMeanStd((obs_dim,))

        self._init_adam()

    # init helpers
    def _init_network(self, sizes: List[int]):
        """Orthogonal init (Engstrom et al. 2020 #4)."""
        weights = []
        biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            a = self.rng.randn(max(fan_in, fan_out), max(fan_in, fan_out))
            q, _ = np.linalg.qr(a)
            w = q[:fan_in, :fan_out]
            if i < len(sizes) - 2:
                w *= np.sqrt(2.0)
            else:
                w *= 0.01
            weights.append(w.astype(np.float64))
            biases.append(np.zeros(fan_out, dtype=np.float64))
        return weights, biases

    def _init_adam(self):
        self.adam_t = 0
        self.adam_m = {}
        self.adam_v = {}
        for prefix, (weights, biases) in [
            ("actor", (self.actor_weights, self.actor_biases)),
            ("critic", (self.critic_weights, self.critic_biases)),
        ]:
            for i, (w, b) in enumerate(zip(weights, biases)):
                self.adam_m[f"{prefix}_w{i}"] = np.zeros_like(w)
                self.adam_v[f"{prefix}_w{i}"] = np.zeros_like(w)
                self.adam_m[f"{prefix}_b{i}"] = np.zeros_like(b)
                self.adam_v[f"{prefix}_b{i}"] = np.zeros_like(b)
        self.adam_m["log_std"] = np.zeros_like(self.log_std)
        self.adam_v["log_std"] = np.zeros_like(self.log_std)

    #observation normalization
    def normalize_obs_array(self, obs: np.ndarray) -> np.ndarray:
        """Apply current running statistics to a single obs (no update)."""
        if not self.normalize_obs:
            return obs
        sigma = np.sqrt(self.obs_rms.var + 1e-8)
        return (obs - self.obs_rms.mean) / sigma

    def update_obs_rms(self, obs: np.ndarray) -> None:
        if self.normalize_obs:
            self.obs_rms.update(obs.reshape(1, -1))

    # forward passes 
    def _forward_actor(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through actor. Returns pre-tanh mean."""
        h = obs.astype(np.float64)
        for i in range(len(self.actor_weights) - 1):
            h = h @ self.actor_weights[i] + self.actor_biases[i]
            h = np.maximum(0, h)
        mean = h @ self.actor_weights[-1] + self.actor_biases[-1]
        return mean

    def _forward_critic(self, obs: np.ndarray) -> np.ndarray:
        h = obs.astype(np.float64)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        for i in range(len(self.critic_weights) - 1):
            h = h @ self.critic_weights[i] + self.critic_biases[i]
            h = np.maximum(0, h)
        val = h @ self.critic_weights[-1] + self.critic_biases[-1]
        return val.reshape(-1)

    def get_value(self, obs: np.ndarray) -> float:
        """V(s) for a single (already-normalised) observation."""
        return float(self._forward_critic(obs.reshape(1, -1))[0])

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Sample an action from the policy, given an already-normalised observation.

        Returns (action, log_prob, value, pre_tanh):
            action    : tanh-squashed in (-1, 1)^d
            log_prob  : log p(action) under the squashed Gaussian
            value     : V(s)
            pre_tanh  : the actual sampled pre-tanh value (Bug 1 fix)
        """
        mean = self._forward_actor(obs.reshape(1, -1)).flatten()
        std = np.exp(self.log_std)
        value = float(self._forward_critic(obs.reshape(1, -1))[0])

        if deterministic:
            # mean is the pre-tanh - no noise added.
            return np.tanh(mean), 0.0, value, mean.copy()

        noise = np.random.randn(self.act_dim) * std
        pre_tanh = mean + noise
        action = np.tanh(pre_tanh)

        log_prob = -0.5 * np.sum(((pre_tanh - mean) / (std + 1e-8)) ** 2)
        log_prob -= 0.5 * self.act_dim * np.log(2 * np.pi)
        log_prob -= np.sum(np.log(std + 1e-8))
        log_prob -= np.sum(np.log(1.0 - action ** 2 + 1e-6))

        return action, float(log_prob), value, pre_tanh

    def evaluate_actions(self, obs_batch: np.ndarray,actions_pretanh: np.ndarray):
        """Evaluate log probs and values for a batch (used at debug time)."""
        means = self._forward_actor(obs_batch)
        std = np.exp(self.log_std)
        values = self._forward_critic(obs_batch).flatten()

        diff = (actions_pretanh - means) / (std + 1e-8)
        log_probs = -0.5 * np.sum(diff ** 2, axis=1)
        log_probs -= 0.5 * self.act_dim * np.log(2 * np.pi)
        log_probs -= np.sum(np.log(std + 1e-8))
        tanh_a = np.tanh(actions_pretanh)
        log_probs -= np.sum(np.log(1.0 - tanh_a ** 2 + 1e-6), axis=1)

        # squashed-policy entropy estimate - include the MC estimate of the tanh log-det-jacobian.
        base_entropy = (0.5 * self.act_dim * (1.0 + np.log(2 * np.pi))+ np.sum(np.log(std + 1e-8)))
        log_det_jac = np.sum(np.log(1.0 - tanh_a ** 2 + 1e-6), axis=1)
        entropy = base_entropy + float(np.mean(log_det_jac))
        return log_probs, values, entropy

    #forward+backward with caching for Adam updates
    def _forward_actor_cached(self, obs: np.ndarray):
        h = np.atleast_2d(obs.astype(np.float64))
        cache = []
        for i in range(len(self.actor_weights) - 1):
            h_in = h
            z = h @ self.actor_weights[i] + self.actor_biases[i]
            h = np.maximum(0.0, z)
            cache.append((h_in, z, h))
        h_in = h
        mean = h @ self.actor_weights[-1] + self.actor_biases[-1]
        cache.append((h_in, mean, mean))
        return mean, cache

    def _forward_critic_cached(self, obs: np.ndarray):
        h = np.atleast_2d(obs.astype(np.float64))
        cache = []
        for i in range(len(self.critic_weights) - 1):
            h_in = h
            z = h @ self.critic_weights[i] + self.critic_biases[i]
            h = np.maximum(0.0, z)
            cache.append((h_in, z, h))
        h_in = h
        out = h @ self.critic_weights[-1] + self.critic_biases[-1]
        cache.append((h_in, out, out))
        return out.flatten(), cache

    def _backward_network(self, weights, biases, cache, grad_output):
        grad_output = np.atleast_2d(grad_output)
        n_layers = len(weights)
        grad_weights = [None] * n_layers
        grad_biases = [None] * n_layers

        h_in, _z, _ = cache[-1]
        grad_weights[-1] = h_in.T @ grad_output
        grad_biases[-1] = grad_output.sum(axis=0)
        grad_h = grad_output @ weights[-1].T

        for i in range(n_layers - 2, -1, -1):
            h_in, z, _ = cache[i]
            grad_z = grad_h * (z > 0).astype(np.float64)
            grad_weights[i] = h_in.T @ grad_z
            grad_biases[i] = grad_z.sum(axis=0)
            grad_h = grad_z @ weights[i].T

        return grad_weights, grad_biases

    # adam updates
    def adam_update(self, param_key: str, param: np.ndarray, grad: np.ndarray, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> np.ndarray:
        """Single Adam parameter update (uses the current self.adam_t)."""
        self.adam_m[param_key] = beta1 * self.adam_m[param_key] + (1 - beta1) * grad
        self.adam_v[param_key] = beta2 * self.adam_v[param_key] + (1 - beta2) * grad ** 2
        m_hat = self.adam_m[param_key] / (1 - beta1 ** self.adam_t)
        v_hat = self.adam_v[param_key] / (1 - beta2 ** self.adam_t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    # save / load with obs-norm folding
    def save(self, directory: str):
        """
        Save weights with obs normalisation FOLDED into the first layer of both actor and critic. The saved network expects
        RAW observations and produces the same outputs as the in-memory (training) network does on normalised observations.

        Math:
            y = ReLU(((x - mu) / sigma) @ W + b)
              = ReLU(x @ W' + b')
            W' = W / sigma[:, None]
            b' = b - (mu / sigma) @ W
        """
        os.makedirs(directory, exist_ok=True)

        if self.normalize_obs:
            sigma = np.sqrt(self.obs_rms.var + 1e-8)
            mu = self.obs_rms.mean
        else:
            sigma = np.ones(self.obs_dim, dtype=np.float64)
            mu = np.zeros(self.obs_dim, dtype=np.float64)

        # actor 
        actor_data = {}
        actor_data["w0"] = self.actor_weights[0] / sigma[:, None]
        actor_data["b0"] = self.actor_biases[0] - (mu / sigma) @ self.actor_weights[0]
        for i in range(1, len(self.actor_weights)):
            actor_data[f"w{i}"] = self.actor_weights[i]
            actor_data[f"b{i}"] = self.actor_biases[i]
        np.savez(os.path.join(directory, "actor.npz"), **actor_data)
        np.save(os.path.join(directory, "log_std.npy"), self.log_std)

        # critic
        critic_data = {}
        critic_data["w0"] = self.critic_weights[0] / sigma[:, None]
        critic_data["b0"] = self.critic_biases[0] - (mu / sigma) @ self.critic_weights[0]
        for i in range(1, len(self.critic_weights)):
            critic_data[f"w{i}"] = self.critic_weights[i]
            critic_data[f"b{i}"] = self.critic_biases[i]
        np.savez(os.path.join(directory, "critic.npz"), **critic_data)

        # obs_rms (for resuming training, not needed at eval time)
        np.savez(
            os.path.join(directory, "obs_rms.npz"),
            mean=mu, var=self.obs_rms.var,
            count=np.array([self.obs_rms.count], dtype=np.float64),
        )

        # config 
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_sizes": [w.shape[1] for w in self.actor_weights[:-1]],
                "obs_normalization_folded": bool(self.normalize_obs),
            }, f, indent=2)

    def load(self, directory: str):
        """
        Load a checkpoint, undoing the first-layer obs-norm folding so training can resume.
        """
        # load obs_rms first so we know how to un-fold the weights.
        rms_path = os.path.join(directory, "obs_rms.npz")
        if os.path.exists(rms_path):
            rms_data = np.load(rms_path)
            self.obs_rms.mean = rms_data["mean"].astype(np.float64)
            self.obs_rms.var = rms_data["var"].astype(np.float64)
            self.obs_rms.count = float(rms_data["count"][0])

        sigma = np.sqrt(self.obs_rms.var + 1e-8) if self.normalize_obs \
            else np.ones(self.obs_dim, dtype=np.float64)
        mu = self.obs_rms.mean if self.normalize_obs \
            else np.zeros(self.obs_dim, dtype=np.float64)

        #  actor: un-fold first layer 
        actor_path = os.path.join(directory, "actor.npz")
        if os.path.exists(actor_path):
            data = np.load(actor_path)
            w0_folded = data["w0"]
            b0_folded = data["b0"]
            self.actor_weights[0] = w0_folded * sigma[:, None]
            # b_orig = b_folded + mu @ w_folded   (see derivation in save())
            self.actor_biases[0] = b0_folded + mu @ w0_folded
            for i in range(1, len(self.actor_weights)):
                self.actor_weights[i] = data[f"w{i}"]
                self.actor_biases[i] = data[f"b{i}"]

        ls_path = os.path.join(directory, "log_std.npy")
        if os.path.exists(ls_path):
            self.log_std = np.load(ls_path)

        #  critic: un-fold first layer 
        critic_path = os.path.join(directory, "critic.npz")
        if os.path.exists(critic_path):
            data = np.load(critic_path)
            w0_folded = data["w0"]
            b0_folded = data["b0"]
            self.critic_weights[0] = w0_folded * sigma[:, None]
            self.critic_biases[0] = b0_folded + mu @ w0_folded
            for i in range(1, len(self.critic_weights)):
                self.critic_weights[i] = data[f"w{i}"]
                self.critic_biases[i] = data[f"b{i}"]


# Rollout buffer ( stores old_values for value-clipping)
class RolloutBuffer:
    """Stores transitions for PPO update with GAE computation."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float64)
        self.actions_pretanh = np.zeros((capacity, act_dim), dtype=np.float64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.values = np.zeros(capacity, dtype=np.float64)
        self.log_probs = np.zeros(capacity, dtype=np.float64)
        self.dones = np.zeros(capacity, dtype=np.float64)
        self.advantages = np.zeros(capacity, dtype=np.float64)
        self.returns = np.zeros(capacity, dtype=np.float64)
        self.ptr = 0

    def add(self, obs, action_pretanh, reward, value, log_prob, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions_pretanh[i] = action_pretanh
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self.ptr += 1

    @property
    def full(self):
        return self.ptr >= self.capacity

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE (Schulman et al. 2016)."""
        n = self.ptr
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Yield random minibatches for PPO epochs (now includes old_values)."""
        n = self.ptr
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield (
                self.obs[idx],
                self.actions_pretanh[idx],
                self.log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
                self.values[idx],
            )

    def reset(self):
        self.ptr = 0

class TrainingNavEnvironment:
    """
    Navigation environment with curriculum-controlled obstacle generation, per-episode start/goal randomisation, and dense reward function.

    Reward components per step:
        r_step                                  : -0.01  (alive cost)
        r_progress * (Δd_goal / d_initial)      : ~ +1 / episode total
        r_align    * cos(v, goal_dir)           : alignment bonus
        r_smooth   * ||a_t - a_{t-1}||          : action-jerk penalty (skips t=1)
        r_clearance * quadratic well            : - near-obstacle barrier
        r_wall     * quadratic well             : - near-wall barrier
        r_goal      on success                  : sparse terminal +
        r_collision on collision                : sparse terminal -
        r_timeout   on timeout                  : sparse terminal -
    """

    def __init__(self, hp: Dict, rng: np.random.RandomState):
        self.hp = hp
        self.rng = rng
        self.curriculum_stage = 0
        
        # initial defaults, overwritten in reset()
        self.ws_lo = -hp.get("workspace_size_max", 3.0)
        self.ws_hi = hp.get("workspace_size_max", 3.0)

        self.goal = np.array([2.0, 2.0, 2.0])
        self.agent_pos = np.zeros(3)
        self.initial_d_goal = 1.0

        self._reset_internals()

    def _reset_internals(self):
        self.obstacles = []
        self.columns = [] 
        self.chaser_idx = -1
        self.step_count = 0
        self.done = False
        self.prev_action = np.zeros(3)
        self.prev_d_goal = None
        self.field_engine = None

    def _build_field_engine(self):
        engine = SuperharmonicFieldEngine(
            goal_position=self.goal, a_att=0.1, a_rep=1.0, n_power=2.0,
            danger_distance=1.0,
            workspace_lo=self.ws_lo, workspace_hi=self.ws_hi,
            a_wall=0.5, wall_power=2.0, wall_danger=0.8,
        )
        for obs in self.obstacles:
            engine.add_obstacle(obs["pos"], obs["radius"], obs.get("vel"))
        # append RectangularColumn objects directly to engine.obstacles
        # so the field-only observation (build_escape_observation) sees
        # their potential and gradient.
        for col in self.columns:
            engine.obstacles.append(col)
        return engine

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage

    def _sample_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        min_dist = self.hp["min_start_goal_distance"]
        for _ in range(50):
            goal = self.rng.uniform(self.ws_lo + 0.6, self.ws_hi - 0.6, size=3)
            start = self.rng.uniform(self.ws_lo + 0.4, self.ws_hi - 0.4, size=3)
            if np.linalg.norm(goal - start) >= min_dist:
                return start, goal
        goal = np.array([self.ws_hi - 1.0, self.ws_hi - 1.0, self.ws_hi - 1.0])
        start = np.array([self.ws_lo + 1.0, self.ws_lo + 1.0, self.ws_lo + 1.0])
        return start, goal

    def reset(self) -> np.ndarray:
        self._reset_internals()

        # randomize workspace boundaries for this episode
        min_size = self.hp.get("workspace_size_min", 2.5)
        max_size = self.hp.get("workspace_size_max", 4.0)
        half_size = self.rng.uniform(min_size, max_size)
        self.ws_lo = -half_size
        self.ws_hi = half_size

        self.agent_pos, self.goal = self._sample_start_goal()
        self.initial_d_goal = max(np.linalg.norm(self.agent_pos - self.goal), 1.0)

        # column spawn probabilities for current stage
        hp_col_p1 = self.hp.get("column_prob_stage1", 0.0)
        hp_col_p2 = self.hp.get("column_prob_stage2", 0.0)

        self.obstacles = []
        self.chaser_idx = -1

        if self.curriculum_stage >= 1:
            n_obs = self.rng.randint(1, 4)
            for _ in range(n_obs):
                pos = self._sample_obstacle_position()
                radius = self.rng.uniform(0.20, 0.45)
                speed = self.rng.uniform(0.05, 0.20)
                direction = self.rng.randn(3)
                direction /= max(np.linalg.norm(direction), 1e-6)
                vel = direction * speed
                self.obstacles.append({"pos": pos, "radius": radius, "vel": vel})

        if self.curriculum_stage >= 2:
            n_extra = self.rng.randint(2, 5)
            for _ in range(n_extra):
                pos = self._sample_obstacle_position()
                radius = self.rng.uniform(0.25, 0.40)
                speed = self.rng.uniform(0.20, 0.45)
                direction = self.rng.randn(3)
                direction /= max(np.linalg.norm(direction), 1e-6)
                vel = direction * speed
                self.obstacles.append({"pos": pos, "radius": radius, "vel": vel})

            if self.rng.rand() < 0.5:
                offset = self.rng.randn(3)
                offset = offset / max(np.linalg.norm(offset), 1e-6)
                chase_pos = self.agent_pos + offset * self.rng.uniform(1.5, 2.5)
                chase_pos = np.clip(chase_pos, self.ws_lo + 0.5, self.ws_hi - 0.5)
                self.obstacles.append({
                    "pos": chase_pos.copy(), "radius": 0.35, "vel": np.zeros(3),
                })
                self.chaser_idx = len(self.obstacles) - 1

        # probabilistically spawn floor-to-ceiling
        # RectangularColumns. Static. Inserted BEFORE _build_field_engine
        # so the engine.obstacles list contains the live column references.
        if self.curriculum_stage >= 1 and self.rng.rand() < hp_col_p1:
            col = self._spawn_column()
            if col is not None:
                self.columns.append(col)
        if self.curriculum_stage >= 2 and self.rng.rand() < hp_col_p2:
            col = self._spawn_column()
            if col is not None:
                self.columns.append(col)

        self.field_engine = self._build_field_engine()
        self.prev_d_goal = np.linalg.norm(self.agent_pos - self.goal)
        self.max_steps = 300 if self.curriculum_stage < 2 else 400

        return self._get_obs()

    def _sample_obstacle_position(self) -> np.ndarray:
        for _ in range(20):
            pos = self.rng.uniform(self.ws_lo + 0.5, self.ws_hi - 0.5, size=3)
            if (np.linalg.norm(pos - self.agent_pos) > 0.8 and np.linalg.norm(pos - self.goal) > 0.8):
                return pos
        return self.rng.uniform(self.ws_lo + 0.5, self.ws_hi - 0.5, size=3)

    def _spawn_column(self) -> Optional[RectangularColumn]:
        """
        sample a floor-to-ceiling RectangularColumn whose footprint is
        at least 0.8 m clear of the start and goal. Returns None if no
        valid placement was found within 20 tries (caller silently skips).
        """
        for _ in range(20):
            cx, cy = self.rng.uniform(self.ws_lo + 0.6, self.ws_hi - 0.6, size=2)
            hx = self.rng.uniform(0.25, 0.45)
            hy = self.rng.uniform(0.25, 0.45)
            # ensure footprint stays inside workspace
            if (cx - hx < self.ws_lo + 0.2 or cx + hx > self.ws_hi - 0.2 or cy - hy < self.ws_lo + 0.2 or cy + hy > self.ws_hi - 0.2):
                continue
            # surface_distance check against start and goal (xy-only;
            # floor-to-ceiling so z is irrelevant).
            tmp = RectangularColumn(
                center_xy=np.array([cx, cy], dtype=float),
                half_extents_xy=np.array([hx, hy], dtype=float),
                z_lo=self.ws_lo, z_hi=self.ws_hi,
                velocity_xy=None, label='col',
            )
            if (tmp.surface_distance(self.agent_pos) > 0.8 and tmp.surface_distance(self.goal) > 0.8):
                return tmp
        return None

    def _get_obs(self) -> np.ndarray:
        return build_escape_observation(
            agent_pos=self.agent_pos,
            field_engine=self.field_engine,
            step_count=self.step_count,
            max_episode_steps=self.max_steps,
            initial_d_goal=self.initial_d_goal,
            ws_lo=self.ws_lo,
            ws_hi=self.ws_hi,
        )

    def step(self, action: np.ndarray):
        """Execute action ([-1,1]^3 -> velocity). v5 reward function."""
        hp = self.hp
        dt = hp["dt"]

        velocity = action * hp["max_velocity"]
        self.agent_pos = self.agent_pos + velocity * dt
        self.agent_pos = np.clip(self.agent_pos, self.ws_lo, self.ws_hi)
        self.step_count += 1

        # update chaser
        if 0 <= self.chaser_idx < len(self.obstacles):
            chaser = self.obstacles[self.chaser_idx]
            chase_vec = self.agent_pos - chaser["pos"]
            cd = np.linalg.norm(chase_vec)
            if cd > 1e-6:
                chaser["vel"] = (chase_vec / cd) * 0.35

        # update obstacle positions
        for i, obs_dict in enumerate(self.obstacles):
            if obs_dict["vel"] is not None:
                obs_dict["pos"] = obs_dict["pos"] + obs_dict["vel"] * dt
                if i != self.chaser_idx:
                    for d in range(3):
                        lo = self.ws_lo + obs_dict["radius"]
                        hi = self.ws_hi - obs_dict["radius"]
                        if obs_dict["pos"][d] <= lo or obs_dict["pos"][d] >= hi:
                            obs_dict["vel"][d] *= -1
                            obs_dict["pos"][d] = np.clip(obs_dict["pos"][d], lo, hi)
                else:
                    obs_dict["pos"] = np.clip(
                        obs_dict["pos"],
                        self.ws_lo + obs_dict["radius"],
                        self.ws_hi - obs_dict["radius"],
                    )

        # sync field engine obstacles
        for i, obs_dict in enumerate(self.obstacles):
            self.field_engine.obstacles[i].position = obs_dict["pos"].copy()
            if obs_dict["vel"] is not None:
                self.field_engine.obstacles[i].velocity = obs_dict["vel"].copy()

        # termination 
        d_goal = np.linalg.norm(self.agent_pos - self.goal)
        min_clearance = float("inf")
        collision = False
        for obs_dict in self.obstacles:
            d = np.linalg.norm(self.agent_pos - obs_dict["pos"]) - obs_dict["radius"]
            min_clearance = min(min_clearance, d)
            if d < hp["collision_threshold"]:
                collision = True
        # true box surface distance for columns instead of the
        # spherical fallback. Fixes false collisions off faces and missed
        # collisions at corners. Also corrects the clearance reward.
        for col in self.columns:
            d = _obstacle_surface_distance(col, self.agent_pos)
            min_clearance = min(min_clearance, d)
            if d < hp["collision_threshold"]:
                collision = True

        wall_d = np.minimum(self.agent_pos - self.ws_lo, self.ws_hi - self.agent_pos)
        wall_clearance = float(wall_d.min())
        wall_collision = wall_clearance < 0.1
        goal_reached = d_goal < hp["goal_tolerance"]
        timeout = self.step_count >= self.max_steps

        # reward
        reward = hp["r_step"]

        # progress, normalised by initial distance.
        progress = (self.prev_d_goal - d_goal) / self.initial_d_goal
        reward += progress * hp["r_progress_scale"]

        # smoothness, but skip step 1 where prev_action is meaningless.
        if self.step_count > 1:
            action_diff = np.linalg.norm(action - self.prev_action)
            reward -= hp["r_smooth_weight"] * action_diff

        # terminal rewards
        if goal_reached:
            reward += hp["r_goal"]
        if collision or wall_collision:
            reward += hp["r_collision"]
        if timeout and not goal_reached:
            reward += hp["r_timeout"]

        self.done = goal_reached or collision or wall_collision or timeout
        self.prev_d_goal = d_goal
        self.prev_action = action.copy()

        info = {
            "success": goal_reached,
            "collision": collision or wall_collision,
            "d_goal": d_goal,
            "min_clearance": min_clearance,
            "steps": self.step_count,
        }
        return self._get_obs(), reward, self.done, info


#ppo update with analytical gradients and value clipping
def _ppo_minibatch_update(agent: PPOActorCritic, obs_b, act_b, old_log_probs, returns, advantages, old_values, clip_eps, clip_eps_vf, vf_coef, entropy_coef, max_grad_norm, lr):
    """One minibatch update with analytical gradients and value clipping."""
    N = obs_b.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    # increment Adam timestep PER PARAMETER UPDATE.
    agent.adam_t += 1

    adv = advantages.copy()
    if N > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    means, actor_cache = agent._forward_actor_cached(obs_b)
    values_2d, critic_cache = agent._forward_critic_cached(obs_b)
    values = values_2d.flatten()

    std = np.exp(agent.log_std)
    inv_std2 = 1.0 / (std ** 2 + 1e-12)

    diff = act_b - means
    quad = -0.5 * np.sum(diff ** 2 * inv_std2, axis=1)
    log_norm = -0.5 * agent.act_dim * np.log(2 * np.pi) - np.sum(np.log(std + 1e-8))
    tanh_a = np.tanh(act_b)
    tanh_corr = -np.sum(np.log(1.0 - tanh_a ** 2 + 1e-6), axis=1)
    new_log_probs = quad + log_norm + tanh_corr

    # policy loss (clipped surrogate)
    log_ratio = new_log_probs - old_log_probs
    ratio = np.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_per_ex = -np.minimum(surr1, surr2)
    policy_loss = float(np.mean(policy_per_ex))

    unclipped_is_min = (surr1 <= surr2)
    grad_logp = np.where(unclipped_is_min, -ratio * adv, 0.0) / N

    grad_means = grad_logp[:, None] * (diff * inv_std2)
    grad_log_std = np.sum(
        grad_logp[:, None] * (diff ** 2 * inv_std2 - 1.0), axis=0
    )
    grad_log_std += -entropy_coef * np.ones_like(agent.log_std)

    # value loss (with clipping)
    v_clipped = old_values + np.clip(values - old_values, -clip_eps_vf, clip_eps_vf)
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    unclipped_won = loss_unclipped >= loss_clipped
    value_loss = float(np.mean(np.maximum(loss_unclipped, loss_clipped)))

    in_clip_range = np.abs(values - old_values) <= clip_eps_vf
    # gradient is 2(v-R)/N when (a) unclipped won, OR (b) clipped won and
    # we are inside the clip range (where v_clipped == values, so the
    # derivative chains through trivially), zero when clipped won and
    # outside the range (constant w.r.t. v).
    active = unclipped_won | in_clip_range
    grad_values = np.where(active, 2.0 * (values - returns) / N, 0.0) * vf_coef

    # backprop 
    grad_actor_w, grad_actor_b = agent._backward_network(agent.actor_weights, agent.actor_biases, actor_cache, grad_means)
    grad_critic_w, grad_critic_b = agent._backward_network(agent.critic_weights, agent.critic_biases, critic_cache, grad_values.reshape(-1, 1))

    # gradient norm clipping
    all_grads = (grad_actor_w + grad_actor_b + grad_critic_w + grad_critic_b + [grad_log_std])
    total_norm = float(np.sqrt(sum(np.sum(g ** 2) for g in all_grads)))
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for lst in (grad_actor_w, grad_actor_b, grad_critic_w, grad_critic_b):
            for j in range(len(lst)):
                lst[j] = lst[j] * clip_coef
        grad_log_std = grad_log_std * clip_coef

    # adam steps 
    for i in range(len(agent.actor_weights)):
        agent.actor_weights[i] = agent.adam_update(f"actor_w{i}", agent.actor_weights[i], grad_actor_w[i], lr)
        agent.actor_biases[i] = agent.adam_update(f"actor_b{i}", agent.actor_biases[i], grad_actor_b[i], lr)
    for i in range(len(agent.critic_weights)):
        agent.critic_weights[i] = agent.adam_update(f"critic_w{i}", agent.critic_weights[i], grad_critic_w[i], lr)
        agent.critic_biases[i] = agent.adam_update(f"critic_b{i}", agent.critic_biases[i], grad_critic_b[i], lr)
    agent.log_std = agent.adam_update("log_std", agent.log_std, grad_log_std, lr)

    # diagnostics
    # squashed-policy entropy estimate.
    base_entropy = (0.5 * agent.act_dim * (1.0 + np.log(2 * np.pi))+ np.sum(np.log(std + 1e-8)))
    log_det_jac = np.sum(np.log(1.0 - tanh_a ** 2 + 1e-6), axis=1)
    entropy = float(base_entropy + np.mean(log_det_jac))

    # Schulman's k3 KL estimator (always >= 0, low variance).
    approx_kl = float(np.mean(np.exp(log_ratio) - 1.0 - log_ratio))

    return policy_loss, value_loss, entropy, approx_kl, N


def ppo_update(agent: PPOActorCritic, buffer: RolloutBuffer, hp: Dict, entropy_coef: float, lr: float):
    """Run PPO update epochs over the rollout buffer."""
    # adam_t is no longer incremented here. It increments
    # inside _ppo_minibatch_update, once per parameter update.
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_updates = 0
    target_kl = hp.get("target_kl", 0.03)
    max_grad_norm = hp.get("max_grad_norm", 0.5)
    vf_coef = hp.get("vf_coef", 0.5)
    clip_eps = hp.get("clip_eps", 0.2)
    clip_eps_vf = hp.get("clip_eps_vf", 0.2)

    early_stop = False
    for epoch in range(hp["n_epochs"]):
        if early_stop:
            break
        for batch in buffer.get_batches(hp["batch_size"]):
            obs_b, act_b, old_lp_b, ret_b, adv_b, old_val_b = batch
            pl, vl, ent, kl, n = _ppo_minibatch_update(agent, obs_b, act_b, old_lp_b, ret_b, adv_b, old_val_b, clip_eps=clip_eps, clip_eps_vf=clip_eps_vf, vf_coef=vf_coef, entropy_coef=entropy_coef,
                                                        max_grad_norm=max_grad_norm, lr=lr)
            total_policy_loss += pl
            total_value_loss += vl
            total_entropy += ent
            total_kl += kl
            n_updates += 1

            # compare positive KL directly (no abs).
            if kl > 1.5 * target_kl:
                early_stop = True
                break

    n = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "entropy": total_entropy / n,
        "approx_kl": total_kl / n,
        "n_minibatches": n_updates,
    }


# evaluation
def evaluate_agent(agent: PPOActorCritic, hp: Dict, n_episodes: int = 20, seed_offset: int = 10000) -> Dict:
    """
    Deterministic evaluation at full curriculum difficulty.
    obs_rms is applied but not updated during eval.
    A single env is reused across episodes.
    """
    successes = 0
    collisions = 0
    total_steps = 0
    total_d_goal = 0.0

    eval_rng = np.random.RandomState(seed_offset)
    env = TrainingNavEnvironment(hp, eval_rng)
    env.set_curriculum_stage(2)

    for ep in range(n_episodes):
        eval_rng.seed(seed_offset + ep)
        obs = env.reset()

        while not env.done:
            obs_n = agent.normalize_obs_array(obs)
            action, _, _, _ = agent.get_action(obs_n, deterministic=True)
            obs, reward, done, info = env.step(action)

        if info["success"]:
            successes += 1
        if info["collision"]:
            collisions += 1
        total_steps += info["steps"]
        total_d_goal += info["d_goal"]

    return {
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_d_goal": total_d_goal / n_episodes,
    }


# main training loop
def train(hp: Dict, save_dir: str, verbose: bool = True):
    """Main PPO training loop with episode-based curriculum (v6)."""
    np.random.seed(hp["seed"])
    rng = np.random.RandomState(hp["seed"])

    agent = PPOActorCritic(
        obs_dim=hp["obs_dim"],
        act_dim=hp["act_dim"],
        hidden_sizes=hp["hidden_sizes"],
        seed=hp["seed"],
        normalize_obs=hp.get("normalize_obs", True),
    )

    buffer = RolloutBuffer(hp["rollout_steps"], hp["obs_dim"], hp["act_dim"])
    env = TrainingNavEnvironment(hp, rng)
    ret_normalizer = ReturnNormalizer(hp["gamma"]) if hp.get("normalize_returns", True) else None

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.json")
    log_entries = []

    total_episodes = hp["total_episodes"]
    episode = 0
    total_steps = 0
    best_eval_sr = -1.0
    last_save_ep = 0
    last_eval_ep = 0
    t_start = time.time()

    if verbose:
        print("=" * 70)
        print("PURE PPO NAVIGATION TRAINING (v6)")
        print(f"  Total episodes target:  {total_episodes:,}")
        print(f"  Curriculum (episodes):  S0 [0..{hp['curriculum_stage1_start']:,}) "
              f"-> S1 [..{hp['curriculum_stage2_start']:,}) -> S2 [..end]")
        print(f"  Network:                hidden={hp['hidden_sizes']}, "
              f"obs_dim={hp['obs_dim']}, act_dim={hp['act_dim']}")
        print(f"  normalize_obs={hp.get('normalize_obs', True)}, "
              f"normalize_returns={hp.get('normalize_returns', True)}")
        print(f"  Save dir:               {save_dir}")
        print("=" * 70)

    #initial reset & first observation 
    obs_raw = env.reset()
    agent.update_obs_rms(obs_raw)
    obs_n = agent.normalize_obs_array(obs_raw)
    ep_reward = 0.0
    ep_steps = 0

    while episode < total_episodes:
        # episode-based curriculum
        if episode < hp["curriculum_stage1_start"]:
            stage = 0
        elif episode < hp["curriculum_stage2_start"]:
            stage = 1
        else:
            stage = 2
        env.set_curriculum_stage(stage)

        # linear schedule
        frac = episode / max(total_episodes, 1)
        entropy_coef = hp["entropy_coef_init"] + frac * (hp["entropy_coef_final"] - hp["entropy_coef_init"])
        lr = hp["lr_init"] + frac * (hp["lr_final"] - hp["lr_init"])

        # collect rollout 
        done = False
        while not buffer.full and episode < total_episodes:
            action, log_prob, value, pre_tanh = agent.get_action(obs_n)
            next_obs_raw, reward, done, info = env.step(action)

            # reward normalisation (Bug 10)
            if ret_normalizer is not None:
                reward_for_buffer = ret_normalizer.normalize(reward, done)
            else:
                reward_for_buffer = reward

            buffer.add(obs_n, pre_tanh, reward_for_buffer, value, log_prob, done)

            ep_reward += reward 
            ep_steps += 1
            total_steps += 1

            obs_raw = next_obs_raw
            agent.update_obs_rms(obs_raw)
            obs_n = agent.normalize_obs_array(obs_raw)

            if done:
                episode += 1

                if verbose and episode % hp["log_interval"] == 0:
                    elapsed = time.time() - t_start
                    print(f"  ep {episode:>5d} | stage {stage} | "
                          f"R={ep_reward:>7.2f} | steps={ep_steps:>4d} | "
                          f"{'SUCCESS' if info['success'] else 'FAIL':>7s} | "
                          f"{'COLL' if info['collision'] else '    ':>4s} | "
                          f"lr={lr:.2e} | ent={entropy_coef:.4f} | "
                          f"total_steps={total_steps:>7,d} | "
                          f"{elapsed:.0f}s", flush=True)

                log_entries.append({
                    "episode": int(episode),
                    "total_steps": int(total_steps),
                    "reward": float(ep_reward),
                    "steps": int(ep_steps),
                    "success": bool(info["success"]),
                    "collision": bool(info["collision"]),
                    "stage": int(stage),
                })

                if episode >= total_episodes:
                    break

                obs_raw = env.reset()
                agent.update_obs_rms(obs_raw)
                obs_n = agent.normalize_obs_array(obs_raw)
                ep_reward = 0.0
                ep_steps = 0

        # PPO update on completed rollout
        last_value = agent.get_value(obs_n) if not done else 0.0
        buffer.compute_gae(last_value, hp["gamma"], hp["gae_lambda"])
        ppo_update(agent, buffer, hp, entropy_coef, lr)
        buffer.reset()

        # periodic evaluation
        if episode - last_eval_ep >= hp["eval_interval"] and episode > 0:
            last_eval_ep = episode
            eval_result = evaluate_agent(agent, hp, n_episodes=hp["eval_episodes"])
            if verbose:
                print(f"  >>> EVAL @ ep {episode}: "
                      f"SR={eval_result['success_rate']:.3f} "
                      f"CR={eval_result['collision_rate']:.3f} "
                      f"avg_steps={eval_result['avg_steps']:.0f} "
                      f"avg_d_goal={eval_result['avg_d_goal']:.2f}", flush=True)

            if eval_result["success_rate"] > best_eval_sr:
                best_eval_sr = eval_result["success_rate"]
                agent.save(os.path.join(save_dir, "best"))
                if verbose:
                    print(f"  >>> New best model saved (SR={best_eval_sr:.3f})",
                          flush=True)

        # periodic checkpoint
        if episode - last_save_ep >= hp["save_interval"] and episode > 0:
            last_save_ep = episode
            agent.save(os.path.join(save_dir, f"checkpoint_{episode}"))

    # final save
    agent.save(os.path.join(save_dir, "final"))
    with open(log_path, "w") as f:
        json.dump({"hyperparams": hp, "log": log_entries}, f, indent=2)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nTraining complete. {episode} eps, {total_steps:,} steps "
              f"in {elapsed:.0f}s")
        print(f"Best eval SR: {best_eval_sr:.3f}")
        print(f"Checkpoints saved to {save_dir}/")

    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train pure PPO navigation baseline for ablation study (v6)"
    )
    parser.add_argument("--total-episodes", type=int, default=HYPERPARAMS["total_episodes"],
                        help="Total episodes to train for (default: 25000)")
    parser.add_argument("--save-dir", type=str, default="ppo_nav_v6_checkpoints")
    parser.add_argument("--seed", type=int, default=HYPERPARAMS["seed"])
    parser.add_argument("--lr", type=float, default=HYPERPARAMS["lr_init"])
    parser.add_argument("--hidden", type=int, nargs="+", default=HYPERPARAMS["hidden_sizes"])
    parser.add_argument("--rollout-steps", type=int, default=HYPERPARAMS["rollout_steps"])
    parser.add_argument("--curriculum-stage1-start", type=int,
                        default=HYPERPARAMS["curriculum_stage1_start"],
                        help="Episode at which curriculum stage 1 begins (default: 2500)")
    parser.add_argument("--curriculum-stage2-start", type=int,
                        default=HYPERPARAMS["curriculum_stage2_start"],
                        help="Episode at which curriculum stage 2 begins (default: 7500)")
    parser.add_argument("--no-normalize-obs", action="store_true",
                        help="Disable observation normalisation (for ablations)")
    parser.add_argument("--no-normalize-returns", action="store_true",
                        help="Disable return normalisation (for ablations)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    hp = HYPERPARAMS.copy()
    hp["total_episodes"] = args.total_episodes
    hp["seed"] = args.seed
    hp["lr_init"] = args.lr
    hp["hidden_sizes"] = args.hidden
    hp["rollout_steps"] = args.rollout_steps
    hp["curriculum_stage1_start"] = args.curriculum_stage1_start
    hp["curriculum_stage2_start"] = args.curriculum_stage2_start
    if args.no_normalize_obs:
        hp["normalize_obs"] = False
    if args.no_normalize_returns:
        hp["normalize_returns"] = False

    agent = train(hp, args.save_dir, verbose=not args.quiet)