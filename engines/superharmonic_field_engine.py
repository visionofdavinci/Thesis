"""
Superharmonic Potential Field Engine for UAV Navigation — WALL-AWARE

This implements time-varying superharmonic artificial potential fields
(TV-SuperAPF) as a complement to the subharmonic fields in
subharmonic_field_engine.py.

Mathematical Foundation:

While subharmonic fields use Gaussian repulsion:
    Phi_sub(x) = a_att * ||x - x_g||^2  +  sum( a_rep * exp(-k * ||x - x_oi||^2) )

superharmonic fields use inverse-power repulsion:
    Phi_super(x) = a_att * ||x - x_g||^2  +  sum( a_rep * (d0 / ri)^n )   for ri < d0
                                                  0                          for ri >= d0

where ri = ||x - x_oi||  and  n > 1.

Wall-Repulsive Potential (THIS IS NEW AND I HAD TO INTEGRATE IT MYSELF SO THAT
THE DRONE DOESN'T HIT THE WALL:

    Phi_wall(x) = sum_d  a_wall * (d_w / d_lo_d)^m  +  a_wall * (d_w / d_hi_d)^m

where:
    d_lo_d = x_d - ws_lo_d    (distance to lower wall on axis d)
    d_hi_d = ws_hi_d - x_d    (distance to upper wall on axis d)
    d_w = wall_danger          (wall influence range, analogous to d0)
    m = wall_power             (repulsion exponent, default 2)
    a_wall = wall strength     (default 0.5)

Only active when d < d_w (compact support, same principle as obstacle repulsion).

The gradient is:
    dPhi_wall/dx_d = -a_wall * m * d_w^m / d_lo_d^(m+1)   (from lower wall)
                     +a_wall * m * d_w^m / d_hi_d^(m+1)    (from upper wall)


Reference: builds on the TV-SAPF framework of arXiv:2402.11601, extending
to the superharmonic regime.
"""

import numpy as np
from typing import List, Optional, Tuple
from engines.subharmonic_field_engine import Obstacle


class SuperharmonicFieldEngine:
    """
    Superharmonic Potential Field Engine with inverse-power repulsion
    and wall-repulsive potential.

    Mirrors SubharmonicFieldEngine (analytical mode) so that
    switching between subharmonic and superharmonic is as simple as swapping
    the engine object.

    Parameters:
    - goal_position : np.ndarray - goal location [x, y, z] or [x, y]
    - a_att : float - attractive field strength.  Phi_att = a_att * ||x - x_g||^2
    - a_rep : float - repulsive field strength.  Phi_rep = a_rep * (d0/r)^n
    - n_power : float - exponent for inverse-power repulsion, must be > 1 for
      superharmonic. n=2 moderate (good default), n=3 aggressive close-range
    - danger_distance : float - maximum range of repulsive influence (d0).
      Beyond d0, Phi_rep = 0
    - epsilon : float - regularization to prevent division by zero at obstacle centres
    - workspace_lo : float or np.ndarray - lower workspace bounds per axis
    - workspace_hi : float or np.ndarray - upper workspace bounds per axis
    - a_wall : float - wall repulsive strength
    - wall_power : float - wall repulsion exponent (>1 for superharmonic)
    - wall_danger : float - wall influence range (distance from wall where
      repulsion activates). Analogous to d0 for obstacles.
    """

    def __init__(self, goal_position: np.ndarray, a_att: float = 0.1,
                 a_rep: float = 1.0, n_power: float = 2.0,
                 danger_distance: float = 1.0, epsilon: float = 1e-4,
                 workspace_lo: float = -3.0, workspace_hi: float = 3.0,
                 a_wall: float = 0.5, wall_power: float = 2.0,
                 wall_danger: float = 0.8):
        """
        Parameters:
        - goal_position : np.ndarray - goal location [x, y, z] or [x, y]
        - a_att : float - attractive field strength
        - a_rep : float - repulsive field strength
        - n_power : float - inverse-power exponent (>1 for superharmonic)
        - danger_distance : float - max range of repulsive influence d0
        - epsilon : float - regularization constant
        - workspace_lo : float or np.ndarray - lower workspace bound(s)
        - workspace_hi : float or np.ndarray - upper workspace bound(s)
        - a_wall : float - wall repulsive field strength
        - wall_power : float - wall repulsion exponent (>1 for superharmonic)
        - wall_danger : float - distance from wall at which repulsion activates
        """
        self.goal_position = np.array(goal_position, dtype=float)
        self.a_att = a_att
        self.a_rep = a_rep
        self.n_power = n_power
        self.danger_distance = danger_distance
        self.epsilon = epsilon
        self.obstacles: List[Obstacle] = []
        self.current_time = 0.0

        # wall parameters
        dim = len(self.goal_position)
        if np.isscalar(workspace_lo):
            self.workspace_lo = np.full(dim, float(workspace_lo))
        else:
            self.workspace_lo = np.array(workspace_lo, dtype=float)
        if np.isscalar(workspace_hi):
            self.workspace_hi = np.full(dim, float(workspace_hi))
        else:
            self.workspace_hi = np.array(workspace_hi, dtype=float)

        self.a_wall = a_wall
        self.wall_power = wall_power
        self.wall_danger = wall_danger

        # compatibility attributes
        self.k_rep = 1.0 / (danger_distance ** 2) if danger_distance > 0 else 1.0
        self.subharmonic_mode = False
        self.superharmonic_mode = True
        self.dt = 0.1

    #workspace bound setters (for curriculum randomization) 

    def set_workspace(self, ws_lo, ws_hi):
        """
        Updates workspace bounds.  Called by the training loop when the
        curriculum randomizes workspace geometry per episode.

        Parameters:
        - ws_lo : float or np.ndarray - lower bound(s)
        - ws_hi : float or np.ndarray - upper bound(s)
        """
        dim = len(self.goal_position)
        if np.isscalar(ws_lo):
            self.workspace_lo = np.full(dim, float(ws_lo))
        else:
            self.workspace_lo = np.array(ws_lo, dtype=float)
        if np.isscalar(ws_hi):
            self.workspace_hi = np.full(dim, float(ws_hi))
        else:
            self.workspace_hi = np.array(ws_hi, dtype=float)

    #obstacle management

    def add_obstacle(self, position, radius, velocity=None):
        """
        Adds an obstacle to the environment.

        Parameters:
        - position : array-like - [x, y, z] or [x, y] position of obstacle centre
        - radius : float - collision radius of the obstacle
        - velocity : array-like, optional - velocity vector for moving obstacles
        Returns: none
        """
        pos = np.array(position, dtype=float)
        vel = np.array(velocity, dtype=float) if velocity is not None else None
        self.obstacles.append(Obstacle(position=pos, radius=radius, velocity=vel))

    def clear_obstacles(self):
        """
        Removes all obstacles from the environment.
        Parameters: none
        Returns: none
        """
        self.obstacles.clear()

    # wall potential helpers 

    def _wall_potential(self, pos: np.ndarray) -> float:
        """
        Computes the wall-repulsive potential.

        Phi_wall(x) = sum_d  a_wall * (d_w / d_lo_d)^m   if d_lo_d < d_w
                            + a_wall * (d_w / d_hi_d)^m   if d_hi_d < d_w

        where d_lo_d = x_d - ws_lo_d, d_hi_d = ws_hi_d - x_d.

        Uses the same inverse-power form as obstacle repulsion so the
        superharmonic property is preserved.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - phi_wall : float - wall repulsive potential
        """
        d_w = self.wall_danger
        m = self.wall_power
        a = self.a_wall
        phi = 0.0

        d_lo = pos - self.workspace_lo  # distance to lower walls
        d_hi = self.workspace_hi - pos  # distance to upper walls

        for d in range(len(pos)):
            dl = max(d_lo[d], self.epsilon)
            dh = max(d_hi[d], self.epsilon)

            if dl < d_w:
                phi += a * (d_w / dl) ** m
            if dh < d_w:
                phi += a * (d_w / dh) ** m

        return phi

    def _wall_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of the wall-repulsive potential.

        Derivation (for lower wall on axis d):
            Phi_lo = a_wall * d_w^m * (x_d - ws_lo_d)^{-m}
            dPhi_lo/dx_d = -a_wall * m * d_w^m * (x_d - ws_lo_d)^{-(m+1)}

        For upper wall on axis d:
            Phi_hi = a_wall * d_w^m * (ws_hi_d - x_d)^{-m}
            dPhi_hi/dx_d = +a_wall * m * d_w^m * (ws_hi_d - x_d)^{-(m+1)}

        The gradient points AWAY from each wall (repulsive), which is what
        we want-> -grad pushes the drone inward.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - grad : np.ndarray - wall gradient vector (same shape as pos)
        """
        d_w = self.wall_danger
        m = self.wall_power
        a = self.a_wall
        grad = np.zeros_like(pos)

        d_lo = pos - self.workspace_lo
        d_hi = self.workspace_hi - pos

        for d in range(len(pos)):
            dl = max(d_lo[d], self.epsilon)
            dh = max(d_hi[d], self.epsilon)

            if dl < d_w:
                # dPhi/dx_d = -a * m * d_w^m / dl^(m+1)  (points away from lo wall = negative)
                grad[d] += -a * m * (d_w ** m) / (dl ** (m + 1))
            if dh < d_w:
                # dPhi/dx_d = +a * m * d_w^m / dh^(m+1)  (points away from hi wall = positive)
                grad[d] += a * m * (d_w ** m) / (dh ** (m + 1))

        return grad

    def _wall_laplacian(self, pos: np.ndarray) -> float:
        """
        Analytical Laplacian of the wall-repulsive potential.

        d^2/dx_d^2 [ d_w^m * d^{-m} ] = m*(m+1) * d_w^m * d^{-(m+2)}

        Each wall face contributes one term (since each only depends on one axis).

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - lap : float - Laplacian of wall potential
        """
        d_w = self.wall_danger
        m = self.wall_power
        a = self.a_wall
        lap = 0.0

        d_lo = pos - self.workspace_lo
        d_hi = self.workspace_hi - pos

        for d in range(len(pos)):
            dl = max(d_lo[d], self.epsilon)
            dh = max(d_hi[d], self.epsilon)

            if dl < d_w:
                lap += a * m * (m + 1) * (d_w ** m) / (dl ** (m + 2))
            if dh < d_w:
                lap += a * m * (m + 1) * (d_w ** m) / (dh ** (m + 2))

        return lap

    # potential Phi(x)

    def compute_potential(self, pos: np.ndarray) -> float:
        """
        Computes the total superharmonic potential at a position,
        including wall repulsion.

        Phi(x) = Phi_att(x) + Phi_obs(x) + Phi_wall(x)

        where:
            Phi_att  = a_att * ||x - x_g||^2
            Phi_obs  = sum_i  a_rep * (d0/ri)^n   for ri < d0
            Phi_wall = sum_d  a_wall * (d_w/d_lo_d)^m + a_wall * (d_w/d_hi_d)^m

        Parameters:
        - pos : np.ndarray - query position, any dimensionality
        Returns:
        - potential : float - total potential at pos
        """
        pos = np.asarray(pos, dtype=float)

        # attractive term: quadratic bowl centred at goal
        diff_goal = pos - self.goal_position
        phi_att = self.a_att * np.dot(diff_goal, diff_goal)

        # repulsive term: inverse-power with compact support
        phi_rep = 0.0
        d0 = self.danger_distance
        n = self.n_power

        for obs in self.obstacles:
            diff = pos - obs.position
            r = max(np.linalg.norm(diff), self.epsilon)
            if r < d0:
                phi_rep += self.a_rep * (d0 / r) ** n

        # wall repulsive term
        phi_wall = self._wall_potential(pos)

        return phi_att + phi_rep + phi_wall

    # gradient grad(Phi(x))

    def compute_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Computes the analytical gradient of the full superharmonic potential,
        including wall repulsion.

        grad(Phi) = grad(Phi_att) + grad(Phi_obs) + grad(Phi_wall)

        Parameters:
        - pos : np.ndarray - query position, any dimensionality
        Returns:
        - gradient : np.ndarray - gradient vector (same shape as pos)
        """
        pos = np.asarray(pos, dtype=float)

        # attractive gradient: points away from goal (so -grad pulls toward goal)
        grad_att = 2.0 * self.a_att * (pos - self.goal_position)

        # repulsive gradient from obstacles
        grad_rep = np.zeros_like(pos)
        d0 = self.danger_distance
        n = self.n_power

        for obs in self.obstacles:
            diff = pos - obs.position
            r = max(np.linalg.norm(diff), self.epsilon)
            if r < d0:
                coeff = -self.a_rep * n * (d0 ** n) / (r ** (n + 2))
                grad_rep += coeff * diff

        # wall repulsive gradient
        grad_wall = self._wall_gradient(pos)

        return grad_att + grad_rep + grad_wall

    # analytical temporal derivative dPhi/dt

    def compute_dphi_dt(self, pos: np.ndarray) -> float:
        """
        Analytical partial temporal derivative of the superharmonic potential.

        dPhi/dt = dPhi_obs/dt + dPhi_wall/dt

        The wall potential only depends on agent position and workspace bounds,
        which are static within an episode, so dPhi_wall/dt = 0.
        Only obstacle terms contribute.

        dPhi_obs/dt = sum_i  a_rep * n * d0^n / ri^(n+2) * (x - x_oi) . v_oi

        Sign convention - same as subharmonic:
            dPhi/dt > 0  ->  obstacle approaching (repulsive potential increasing)
            dPhi/dt < 0  ->  obstacle receding    (repulsive potential decreasing)
            dPhi/dt ~ 0  ->  static or tangential motion

        The sensing range is extended to 2*d0 so obstacles approaching from just
        outside the active repulsion zone are detected before they enter it.

        Parameters:
        - pos : np.ndarray - query position (agent location)
        Returns:
        - dphi_dt : float - time derivative of the total potential at pos
        """
        pos = np.asarray(pos, dtype=float)
        d0 = self.danger_distance
        n = self.n_power
        # sensing range is 2x the potential support radius
        sensing_range = d0 * 2.0
        dphi_dt = 0.0

        for obs in self.obstacles:
            if obs.velocity is None:
                continue
            vel = obs.velocity
            if np.linalg.norm(vel) < 1e-12:
                continue

            diff = pos - obs.position
            r = max(np.linalg.norm(diff), self.epsilon)

            if r < sensing_range:
                coeff = self.a_rep * n * (d0 ** n) / (r ** (n + 2))
                dphi_dt += coeff * np.dot(diff, vel)

        # wall potential is time-independent (workspace bounds are static
        # within an episode), so dPhi_wall/dt = 0.  No contribution.

        return dphi_dt

    # numerical laplacian delta(Phi) (for verification)

    def compute_laplacian(self, pos: np.ndarray, eps: float = 1e-5) -> float:
        """
        Numerical Laplacian through central finite differences.
        Used to verify the superharmonic property: delta(Phi_rep) > 0 for n > 1.

        Parameters:
        - pos : np.ndarray - query position
        - eps : float - finite difference step
        Returns:
        - laplacian : float - delta(Phi) = sum_d d^2(Phi)/dx_d^2
        """
        pos = np.asarray(pos, dtype=float)
        laplacian = 0.0
        phi_center = self.compute_potential(pos)

        for d in range(len(pos)):
            pp = pos.copy(); pp[d] += eps
            pm = pos.copy(); pm[d] -= eps
            laplacian += (self.compute_potential(pp) - 2.0 * phi_center +
                          self.compute_potential(pm)) / (eps ** 2)

        return laplacian

    #analytical laplacian of repulsive term 

    def compute_laplacian_repulsive_analytical(self, pos: np.ndarray) -> float:
        """
        Analytical Laplacian of the repulsive + wall terms.

        For obstacle: delta(Phi_rep) = a_rep * d0^n * n * (n + 2 - dim) / r^(n+2)
        For wall:     delta(Phi_wall) = sum over faces of a_wall * m*(m+1) * d_w^m / d^(m+2)

        In 3D (dim=3):  delta(r^(-n)) = n(n-1)/r^(n+2)
        In 2D (dim=2):  delta(r^(-n)) = n^2/r^(n+2)

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - laplacian : float - analytical Laplacian of the repulsive + wall potential
        """
        pos = np.asarray(pos, dtype=float)
        dim = len(pos)
        d0 = self.danger_distance
        n = self.n_power
        lap = 0.0

        for obs in self.obstacles:
            diff = pos - obs.position
            r = max(np.linalg.norm(diff), self.epsilon)
            if r < d0:
                lap += self.a_rep * (d0 ** n) * n * (n + 2 - dim) / (r ** (n + 2))

        # attractive Laplacian: delta(a_att * r^2) = 2 * a_att * dim
        lap += 2.0 * self.a_att * dim

        # wall Laplacian
        lap += self._wall_laplacian(pos)

        return lap

    #hybrid gradient with logarithmic long-range attraction

    def compute_hybrid_gradient(self, pos: np.ndarray,
                                log_weight: float = 0.3) -> np.ndarray:
        """
        Hybrid gradient combining superharmonic repulsion with logarithmic
        long-range attraction. This solves the "flat region problem" at distance.

        grad(Phi_hybrid) = grad(Phi_super)  +  log_weight * (x - x_g) / ||x - x_g||^2

        The 1/r attraction term decays more slowly than the quadratic gradient,
        ensuring meaningful pull even far from the goal.

        Parameters:
        - pos : np.ndarray - query position
        - log_weight : float - weight of the logarithmic attraction term
        Returns:
        - gradient : np.ndarray - hybrid gradient vector
        """
        grad = self.compute_gradient(pos)

        diff_goal = pos - self.goal_position
        r_goal_sq = np.dot(diff_goal, diff_goal)

        if r_goal_sq > 1e-8:
            grad += log_weight * diff_goal / r_goal_sq

        return grad

    # field features vector (for RL observation)

    def compute_field_features(self, pos: np.ndarray) -> dict:
        """
        Computes all field features at a position, packaged for RL integration.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - features : dict with keys:
            'potential'  : float     - Phi(x) (includes wall potential)
            'gradient'   : ndarray   - grad(Phi(x)) (includes wall gradient)
            'grad_mag'   : float     - ||grad(Phi)||
            'dphi_dt'    : float     - dPhi/dt (analytical)
            'laplacian'  : float     - delta(Phi) (numerical)
        """
        phi = self.compute_potential(pos)
        grad = self.compute_gradient(pos)
        dphi_dt = self.compute_dphi_dt(pos)
        lap = self.compute_laplacian(pos)

        return {
            'potential': phi,
            'gradient': grad,
            'grad_mag': np.linalg.norm(grad),
            'dphi_dt': dphi_dt,
            'laplacian': lap,
        }