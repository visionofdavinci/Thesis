"""
Reference: PPO algorithm from Schulman et al. "Proximal Policy Optimization
Algorithms" (2017), adapted for continuous control with tanh-squashed Gaussian.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import os


#numpy MLP: minimal neural network for actor and critic 

class NumpyMLP:
    """
    Minimal feedforward network with ReLU hidden layers and configurable output.
    Implements forward pass, backpropagation, and gradient clipping.

    Parameters:
    - layer_sizes : list of int - e.g. [obs_dim, 64, 64, act_dim]
    - output_activation : str - 'none', 'tanh', or 'softplus'
    - init_scale : float - weight initialization scale (Xavier-like)
    """

    def __init__(self, layer_sizes: list, output_activation: str = 'none',
                 init_scale: float = 1.0):
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation
        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = init_scale * np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out).astype(np.float64) * std)
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

        # cache for backpropagation
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with ReLU hidden activations.

        Parameters:
        - x : np.ndarray - input features
        Returns:
        - output : np.ndarray - network output
        """
        self._cache['input'] = x.copy()
        self._cache['pre_activations'] = []
        self._cache['activations'] = [x.copy()]

        h = x
        for i in range(len(self.weights) - 1):
            z = h @ self.weights[i] + self.biases[i]
            self._cache['pre_activations'].append(z.copy())
            h = np.maximum(0, z)  # ReLU
            self._cache['activations'].append(h.copy())

        # output layer (no ReLU, uses configured activation)
        z = h @ self.weights[-1] + self.biases[-1]
        self._cache['pre_activations'].append(z.copy())

        if self.output_activation == 'tanh':
            out = np.tanh(z)
        elif self.output_activation == 'softplus':
            out = np.log1p(np.exp(np.clip(z, -20, 20)))
        else:
            out = z

        self._cache['output'] = out.copy()
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[list, list]:
        """
        Backpropagation. Returns (weight_grads, bias_grads).

        Parameters:
        - grad_output : np.ndarray - gradient of loss w.r.t. network output,
          shape (batch, out_dim) or (out_dim,)
        Returns:
        - weight_grads : list of np.ndarray
        - bias_grads : list of np.ndarray
        """
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        # output activation derivative
        z_out = self._cache['pre_activations'][-1]
        if z_out.ndim == 1:
            z_out = z_out.reshape(1, -1)

        if self.output_activation == 'tanh':
            out = self._cache['output']
            if out.ndim == 1:
                out = out.reshape(1, -1)
            dz = grad_output * (1 - out ** 2)
        elif self.output_activation == 'softplus':
            sig = 1.0 / (1.0 + np.exp(-np.clip(z_out, -20, 20)))
            dz = grad_output * sig
        else:
            dz = grad_output

        weight_grads = []
        bias_grads = []
        n_layers = len(self.weights)

        for i in range(n_layers - 1, -1, -1):
            a_prev = self._cache['activations'][i]
            if a_prev.ndim == 1:
                a_prev = a_prev.reshape(1, -1)

            w_grad = a_prev.T @ dz / dz.shape[0]
            b_grad = dz.mean(axis=0)

            weight_grads.insert(0, w_grad)
            bias_grads.insert(0, b_grad)

            if i > 0:
                dz = dz @ self.weights[i].T
                z_prev = self._cache['pre_activations'][i - 1]
                if z_prev.ndim == 1:
                    z_prev = z_prev.reshape(1, -1)
                dz = dz * (z_prev > 0).astype(float)  # ReLU derivative

        return weight_grads, bias_grads

    def get_params(self) -> list:
        """Returns flat parameter list."""
        return [p.copy() for p in self.weights + self.biases]

    def set_params(self, params: list):
        """Sets network parameters from flat list."""
        n = len(self.weights)
        self.weights = [p.copy() for p in params[:n]]
        self.biases = [p.copy() for p in params[n:]]

    def save(self, path: str):
        """Saves weights to npz file."""
        data = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f'w{i}'] = w
            data[f'b{i}'] = b
        np.savez(path, **data)

    def load(self, path: str):
        """Loads weights from npz file."""
        data = np.load(path)
        for i in range(len(self.weights)):
            self.weights[i] = data[f'w{i}']
            self.biases[i] = data[f'b{i}']


# tanh-squashed Gaussian policy for continuous actions
# had to fix: proper tanh squashing with log-det Jacobian correction

class GaussianPolicy:
    """
    Tanh-squashed diagonal Gaussian policy for continuous action spaces.

    The actor network outputs an UNBOUNDED mean mu(s).  Actions are sampled as:
        u ~ N(mu(s), diag(sigma^2))
        a = tanh(u)
    with log-probability corrected by the change-of-variable Jacobian:
        log pi(a|s) = log N(u|mu,sigma) - sum(log(1 - tanh(u)^2))

    This avoids the gradient-killing saturation of the old tanh output activation,
    because the actor's output z is unbounded and the tanh is only applied to the
    sample, keeping gradients alive at the boundary of the action space.

    Parameters:
    - obs_dim : int - observation dimensionality
    - act_dim : int - action dimensionality (3 for 3D velocity correction)
    - hidden_sizes : list of int - hidden layer sizes for the actor MLP
    - init_log_std : float - initial log standard deviation.
      -0.5 gives sigma ~ 0.6 (moderate exploration)
    - use_cholesky : bool - if True, use full Cholesky covariance (for 3D
      correlated exploration).  Adds act_dim*(act_dim-1)/2 = 3 extra params.
    """

    LOG_STD_MIN = -3.0   # sigma >= 0.05  (prevents collapse)
    LOG_STD_MAX = 0.5     # sigma <= 1.65  (prevents explosion)

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_sizes: list = [64, 64],
                 init_log_std: float = -0.5,
                 use_cholesky: bool = False):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_cholesky = use_cholesky

        #tanh is applied to the sample, not the mean.
        self.actor = NumpyMLP(
            [obs_dim] + hidden_sizes + [act_dim],
            output_activation='none',     # <- changed from 'tanh'
            init_scale=0.5,
        )

        # smaller init for last layer to keep initial mean near 0
        last_idx = len(self.actor.weights) - 1
        fan_in = hidden_sizes[-1]
        self.actor.weights[last_idx] *= 0.01
        self.actor.biases[last_idx] *= 0.0

        # log standard deviation (state-independent, learnable)
        self.log_std = np.full(act_dim, init_log_std, dtype=np.float64)

        # Cholesky off-diagonal elements (lower triangular, below diagonal)
        # for act_dim=3: L = [[1,0,0],[l10,1,0],[l20,l21,1]] * diag(sigma)
        self.n_chol_offdiag = act_dim * (act_dim - 1) // 2
        if use_cholesky:
            self.chol_offdiag = np.zeros(self.n_chol_offdiag, dtype=np.float64)
        else:
            self.chol_offdiag = None

    def _get_cholesky_L(self) -> np.ndarray:
        """Build lower-triangular Cholesky factor L such that Sigma = L @ L.T"""
        d = self.act_dim
        std = np.exp(np.clip(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        L = np.diag(std)
        if self.use_cholesky and self.chol_offdiag is not None:
            idx = 0
            for i in range(d):
                for j in range(i):
                    L[i, j] = self.chol_offdiag[idx] * std[i]
                    idx += 1
        return L

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes unbounded mean and std.

        Parameters:
        - obs : np.ndarray - observation vector
        Returns:
        - mean : np.ndarray - unbounded action mean
        - std : np.ndarray - action standard deviation (clamped)
        """
        mean = self.actor.forward(obs)
        std = np.exp(np.clip(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        return mean, std

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Samples action with tanh squashing and computes corrected log probability.

        Parameters:
        - obs : np.ndarray - observation vector
        Returns:
        - action : np.ndarray - squashed action in [-1, 1]^3
        - log_prob : float - corrected log probability
        - pre_tanh : np.ndarray - pre-squash sample u (needed for log_prob recomputation)
        """
        mean, std = self.forward(obs)

        if self.use_cholesky:
            L = self._get_cholesky_L()
            noise = np.random.randn(self.act_dim)
            u = mean + L @ noise
            # log prob under the full covariance Gaussian
            # log N(u|mu, Sigma) = -0.5 * (u-mu)^T Sigma^{-1} (u-mu) - 0.5*log|Sigma| - d/2*log(2pi)
            diff = u - mean
            # Sigma^{-1} = L^{-T} L^{-1}, so solve for z = L^{-1}(u-mu)
            z = np.linalg.solve(L, diff)
            log_prob_gauss = -0.5 * np.sum(z ** 2) - np.sum(np.log(np.abs(np.diag(L)))) \
                             - 0.5 * self.act_dim * np.log(2 * np.pi)
        else:
            noise = np.random.randn(self.act_dim)
            u = mean + std * noise
            # diagonal Gaussian log prob
            log_prob_gauss = -0.5 * np.sum(
                ((u - mean) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi)
            )

        action = np.tanh(u)
        # log |det d(tanh)/du| = sum(log(1 - tanh(u)^2))
        # numerically stable: log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        log_prob = log_prob_gauss - np.sum(
            np.log(np.maximum(1.0 - action ** 2, 1e-6))
        )

        return action, log_prob, u

    def log_prob(self, obs: np.ndarray, action: np.ndarray,
                 pre_tanh: np.ndarray = None) -> float:
        """
        Computes corrected log probability of a given action.

        Parameters:
        - obs : np.ndarray - observation vector
        - action : np.ndarray - squashed action (in [-1,1])
        - pre_tanh : np.ndarray - if available, the pre-tanh sample u
        Returns:
        - log_prob : float
        """
        mean, std = self.forward(obs)

        # recover pre-tanh value if not provided
        if pre_tanh is not None:
            u = pre_tanh
        else:
            # atanh with clipping for numerical safety
            action_clipped = np.clip(action, -0.999, 0.999)
            u = np.arctanh(action_clipped)

        if self.use_cholesky:
            L = self._get_cholesky_L()
            diff = u - mean
            z = np.linalg.solve(L, diff)
            log_prob_gauss = -0.5 * np.sum(z ** 2) - np.sum(np.log(np.abs(np.diag(L)))) \
                             - 0.5 * self.act_dim * np.log(2 * np.pi)
        else:
            log_prob_gauss = -0.5 * np.sum(
                ((u - mean) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi)
            )

        # Jacobian correction
        tanh_u = np.tanh(u)
        log_prob = log_prob_gauss - np.sum(
            np.log(np.maximum(1.0 - tanh_u ** 2, 1e-6))
        )
        return log_prob

    def entropy(self) -> float:
        """
        Entropy of the Gaussian policy (state-independent part).
        For diagonal: H = 0.5 * d * (1 + log(2pi)) + sum(log(sigma))
        Note: this is the entropy of the pre-squash Gaussian (approximate).
        """
        std = np.exp(np.clip(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX))
        if self.use_cholesky:
            L = self._get_cholesky_L()
            return 0.5 * self.act_dim * (1 + np.log(2 * np.pi)) + \
                   np.sum(np.log(np.abs(np.diag(L))))
        else:
            return 0.5 * self.act_dim * (1 + np.log(2 * np.pi)) + np.sum(np.log(std))


# rollout buffer for on-policy data

class RolloutBuffer:
    """
    Storage for on-policy rollout data collected during training episodes.
    Now also stores pre_tanh values for correct log_prob recomputation.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.pre_tanhs = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def add(self, obs, action, pre_tanh, reward, value, log_prob, done):
        """
        Adds a single transition to the buffer.

        Parameters:
        - obs : np.ndarray - observation
        - action : np.ndarray - squashed action taken
        - pre_tanh : np.ndarray - pre-squash sample u
        - reward : float - reward received
        - value : float - critic value estimate
        - log_prob : float - corrected log probability
        - done : bool - whether episode ended
        """
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.pre_tanhs.append(pre_tanh.copy())
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, agent, last_value: float):
        """
        Computes GAE advantages and discounted returns.
        """
        self.advantages, self.returns = agent.compute_gae(
            self.rewards, self.values, self.dones, last_value
        )

    def get(self) -> dict:
        """Returns all stored data as a dictionary."""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'pre_tanhs': self.pre_tanhs,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
            'advantages': self.advantages,
            'returns': self.returns,
        }


#the PPO agent

class PPOAgent:
    """
    Proximal Policy Optimization agent for escape actions.

    Parameters:
    - obs_dim : int - observation space dimension
    - act_dim : int - action space dimension (3 for 3D velocity correction)
    - lr_actor : float - actor learning rate
    - lr_critic : float - critic learning rate
    - gamma : float - discount factor
    - gae_lambda : float - GAE lambda for advantage estimation
    - clip_epsilon : float - PPO clipping parameter
    - entropy_coeff : float - entropy bonus coefficient (encourages exploration)
    - value_coeff : float - value loss coefficient
    - max_grad_norm : float - gradient clipping norm
    - n_epochs : int - number of optimization epochs per update
    - batch_size : int - minibatch size for updates
    - hidden_sizes : list - hidden layer sizes for actor and critic
    - use_cholesky : bool - use Cholesky covariance for correlated exploration
    """

    def __init__(self, obs_dim: int, act_dim: int = 3,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 n_epochs: int = 10, batch_size: int = 64,
                 hidden_sizes: list = None,
                 use_cholesky: bool = False):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # policy network (actor + log_std, optionally Cholesky)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes,
                                     use_cholesky=use_cholesky)

        # value function (critic)
        self.critic = NumpyMLP(
            [obs_dim] + hidden_sizes + [1],
            output_activation='none',
            init_scale=1.0,
        )

        # rollout buffer
        self.buffer = RolloutBuffer()

        # training stats
        self.total_updates = 0
        self.training_log = []

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Selects action for the current observation.

        Parameters:
        - obs : np.ndarray - observation vector
        Returns:
        - action : np.ndarray - 3D velocity correction in [-1, 1]^3
        - log_prob : float - corrected log probability
        - value : float - estimated state value
        - pre_tanh : np.ndarray - pre-squash sample (for buffer storage)
        """
        action, log_prob, pre_tanh = self.policy.sample(obs)

        # single critic forward pass
        val_out = self.critic.forward(obs)
        value = val_out.item() if val_out.ndim == 0 else float(val_out[0])

        return action, log_prob, value, pre_tanh

    def evaluate(self, obs: np.ndarray, action: np.ndarray,
                 pre_tanh: np.ndarray = None) -> Tuple[float, float, float]:
        """
        Evaluates a stored (obs, action) pair.

        Parameters:
        - obs : np.ndarray - observation
        - action : np.ndarray - squashed action taken
        - pre_tanh : np.ndarray - pre-squash sample
        Returns:
        - log_prob : float
        - value : float
        - entropy : float
        """
        log_prob = self.policy.log_prob(obs, action, pre_tanh=pre_tanh)
        val_out = self.critic.forward(obs)
        value = val_out.item() if val_out.ndim == 0 else float(val_out[0])
        entropy = self.policy.entropy()
        return log_prob, value, entropy

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Computes Generalized Advantage Estimation.

        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        returns = np.zeros(T, dtype=np.float64)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self):
        """
        Runs PPO update on collected rollout data.

        Performs n_epochs of minibatch SGD on the clipped policy objective:
            L_clip = max(-ratio * A, -clip(ratio) * A)
        plus value function loss and entropy bonus.
        """
        data = self.buffer.get()
        if len(data['observations']) < self.batch_size:
            self.buffer.clear()
            return {'skipped': True}

        observations = np.array(data['observations'])
        actions = np.array(data['actions'])
        pre_tanhs = np.array(data['pre_tanhs'])
        old_log_probs = np.array(data['log_probs'])
        advantages = np.array(data['advantages'])
        returns = np.array(data['returns'])

        # normalize advantages for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        n_samples = len(observations)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_batches = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                bs = end - start

                pg_loss_sum = 0.0
                vf_loss_sum = 0.0
                entropy_sum = 0.0
                clip_count = 0

                # accumulate gradients over minibatch
                actor_w_grads = [np.zeros_like(w) for w in self.policy.actor.weights]
                actor_b_grads = [np.zeros_like(b) for b in self.policy.actor.biases]
                critic_w_grads = [np.zeros_like(w) for w in self.critic.weights]
                critic_b_grads = [np.zeros_like(b) for b in self.critic.biases]
                log_std_grad = np.zeros_like(self.policy.log_std)
                chol_grad = np.zeros(self.policy.n_chol_offdiag, dtype=np.float64) \
                    if self.policy.use_cholesky else None

                for i in batch_idx:
                    obs_i = observations[i]
                    act_i = actions[i]
                    pt_i = pre_tanhs[i]
                    old_lp_i = old_log_probs[i]
                    adv_i = advantages[i]
                    ret_i = returns[i]

                    # forward pass: get new log_prob, value, entropy
                    new_lp_i, val_i, ent_i = self.evaluate(obs_i, act_i, pre_tanh=pt_i)

                    # PPO clipped objective
                    ratio = np.exp(np.clip(new_lp_i - old_lp_i, -20, 20))
                    clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon,
                                            1 + self.clip_epsilon)
                    pg_loss1 = -adv_i * ratio
                    pg_loss2 = -adv_i * clipped_ratio
                    pg_loss = max(pg_loss1, pg_loss2)

                    # check which branch of max() is active, not whether
                    # ratio is inside the clip range.  Gradient flows only through
                    # the active (larger) branch; the clipped branch has zero
                    # gradient w.r.t. log_prob because clipping kills the path.
                    if pg_loss1 >= pg_loss2:
                        # unclipped branch is active - full gradient
                        dloss_dlogp = -adv_i * ratio
                    else:
                        # clipped branch is active - gradient is zero
                        dloss_dlogp = 0.0

                    clipped = (abs(ratio - 1.0) > self.clip_epsilon)
                    clip_count += int(clipped)

                    # value loss (MSE)
                    vf_loss = 0.5 * (val_i - ret_i) ** 2

                    pg_loss_sum += pg_loss
                    vf_loss_sum += vf_loss
                    entropy_sum += ent_i

                    # backprop for actor
                    # We need d(loss)/d(mean) through the chain:
                    #   loss -> log_prob -> u (pre_tanh) -> mean
                    # where log_prob = log N(u|mean, sigma) - log|det J_tanh|
                    # The Jacobian correction depends on u, not on mean, so:
                    #   d(log_prob)/d(mean) = d(log N(u|mean,sigma))/d(mean)
                    #                       = (u - mean) / sigma^2   (diagonal case)

                    mean_i, std_i = self.policy.forward(obs_i)
                    u_i = pt_i  # pre-tanh sample

                    if self.policy.use_cholesky:
                        L = self.policy._get_cholesky_L()
                        diff = u_i - mean_i
                        z = np.linalg.solve(L, diff)
                        # d(log_prob)/d(mean) = Sigma^{-1} (u - mean) = L^{-T} z
                        dlogp_dmean = np.linalg.solve(L.T, z)
                    else:
                        dlogp_dmean = (u_i - mean_i) / (std_i ** 2)

                    # d(log_prob)/d(log_std) = (u - mean)^2/sigma^2 - 1
                    dlogp_dlogstd = ((u_i - mean_i) ** 2 / (std_i ** 2) - 1.0)

                    # entropy gradient: d(entropy)/d(log_std) = 1
                    dent_dlogstd = np.ones_like(self.policy.log_std)

                    # policy loss gradient flows through mean.
                    # entropy gradient flows ONLY through log_std (not mean!).
                    dmean = dloss_dlogp * dlogp_dmean
                    dlogstd = dloss_dlogp * dlogp_dlogstd - self.entropy_coeff * dent_dlogstd

                    # actor network backprop
                    aw, ab = self.policy.actor.backward(dmean.reshape(1, -1))
                    for j in range(len(actor_w_grads)):
                        actor_w_grads[j] += aw[j]
                        actor_b_grads[j] += ab[j]
                    log_std_grad += dlogstd

                    #backprop for critic 
                    dval = self.value_coeff * (val_i - ret_i)
                    cw, cb = self.critic.backward(np.array([[dval]]))
                    for j in range(len(critic_w_grads)):
                        critic_w_grads[j] += cw[j]
                        critic_b_grads[j] += cb[j]

                # average gradients
                for j in range(len(self.policy.actor.weights)):
                    actor_w_grads[j] /= bs
                    actor_b_grads[j] /= bs
                for j in range(len(self.critic.weights)):
                    critic_w_grads[j] /= bs
                    critic_b_grads[j] /= bs
                log_std_grad /= bs

                all_actor_grads = actor_w_grads + actor_b_grads + [log_std_grad]
                global_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_actor_grads))
                if global_norm > self.max_grad_norm:
                    scale = self.max_grad_norm / global_norm
                    for j in range(len(actor_w_grads)):
                        actor_w_grads[j] *= scale
                        actor_b_grads[j] *= scale
                    log_std_grad *= scale

                # global gradient norm clipping for critic
                all_critic_grads = critic_w_grads + critic_b_grads
                critic_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_critic_grads))
                if critic_norm > self.max_grad_norm:
                    scale = self.max_grad_norm / critic_norm
                    for j in range(len(critic_w_grads)):
                        critic_w_grads[j] *= scale
                        critic_b_grads[j] *= scale

                # apply gradients
                for j in range(len(self.policy.actor.weights)):
                    self.policy.actor.weights[j] -= self.lr_actor * actor_w_grads[j]
                    self.policy.actor.biases[j] -= self.lr_actor * actor_b_grads[j]

                self.policy.log_std -= self.lr_actor * log_std_grad
                self.policy.log_std = np.clip(
                    self.policy.log_std,
                    GaussianPolicy.LOG_STD_MIN,
                    GaussianPolicy.LOG_STD_MAX,
                )

                for j in range(len(self.critic.weights)):
                    self.critic.weights[j] -= self.lr_critic * critic_w_grads[j]
                    self.critic.biases[j] -= self.lr_critic * critic_b_grads[j]

                total_pg_loss += pg_loss_sum / bs
                total_vf_loss += vf_loss_sum / bs
                total_entropy += entropy_sum / bs
                total_clip_frac += clip_count / bs
                n_batches += 1

        self.total_updates += 1
        self.buffer.clear()

        metrics = {
            'pg_loss': total_pg_loss / max(n_batches, 1),
            'vf_loss': total_vf_loss / max(n_batches, 1),
            'entropy': total_entropy / max(n_batches, 1),
            'clip_frac': total_clip_frac / max(n_batches, 1),
            'n_samples': n_samples,
            'update_idx': self.total_updates,
        }
        self.training_log.append(metrics)
        return metrics

    def save(self, directory: str):
        """Saves actor, critic, log_std, and optional cholesky to files."""
        os.makedirs(directory, exist_ok=True)
        self.policy.actor.save(os.path.join(directory, 'actor.npz'))
        self.critic.save(os.path.join(directory, 'critic.npz'))
        np.save(os.path.join(directory, 'log_std.npy'), self.policy.log_std)
        if self.policy.use_cholesky and self.policy.chol_offdiag is not None:
            np.save(os.path.join(directory, 'chol_offdiag.npy'),
                    self.policy.chol_offdiag)

        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump({
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
                'total_updates': self.total_updates,
                'use_cholesky': self.policy.use_cholesky,
            }, f, indent=2)

    def load(self, directory: str):
        """Loads saved agent weights from a directory."""
        self.policy.actor.load(os.path.join(directory, 'actor.npz'))
        self.critic.load(os.path.join(directory, 'critic.npz'))
        self.policy.log_std = np.load(os.path.join(directory, 'log_std.npy'))
        chol_path = os.path.join(directory, 'chol_offdiag.npy')
        if self.policy.use_cholesky and os.path.exists(chol_path):
            self.policy.chol_offdiag = np.load(chol_path)