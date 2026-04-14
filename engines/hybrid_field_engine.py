"""
Hybrid Potential Field Engine for UAV Navigation

This engine switches between subharmonic and superharmonic fields
based on the current threat level (dPhi/dt). The switching uses
hysteresis (Schmitt trigger) to prevent rapid oscillation between modes.

Decision logic:
    dPhi/dt < switch_threshold                  ->  subharmonic  (smooth navigation)
    dPhi/dt >= switch_threshold                  ->  superharmonic (aggressive avoidance)
    switch_threshold - hysteresis < dPhi/dt      ->  stay in current mode (dead band)
"""

import numpy as np
from engines.superharmonic_field_engine import SuperharmonicFieldEngine
from engines.subharmonic_field_engine import SubharmonicFieldEngine


class HybridFieldEngine:
    """
    Hybrid engine that switches between subharmonic and superharmonic fields
    based on the current threat level (dPhi/dt).

    Parameters:
    - sub_engine : SubharmonicFieldEngine - subharmonic engine (analytical mode)
    - super_engine : SuperharmonicFieldEngine - superharmonic engine
    - switch_threshold : float - dPhi/dt value above which we switch to superharmonic
    - hysteresis : float - dead band width to prevent rapid switching.
      Switch back to subharmonic only when dPhi/dt < switch_threshold - hysteresis
    - blend_width : float - if > 0, use smooth sigmoid blending in the
      transition band instead of hard switching. 0 = hard switch.
    """

    def __init__(self, sub_engine: SubharmonicFieldEngine,
                 super_engine: SuperharmonicFieldEngine,
                 switch_threshold: float = 0.1,
                 hysteresis: float = 0.03,
                 blend_width: float = 0.0):
        """
        Parameters:
        - sub_engine : SubharmonicFieldEngine - subharmonic engine
        - super_engine : SuperharmonicFieldEngine - superharmonic engine
        - switch_threshold : float - dPhi/dt threshold for switching
        - hysteresis : float - dead band for switch-back
        - blend_width : float - sigmoid blend width (0 = hard switch)
        """
        self.sub = sub_engine
        self.sup = super_engine
        self.switch_threshold = switch_threshold
        self.hysteresis = hysteresis
        self.blend_width = blend_width

        #state tracking
        self.active_mode = 'subharmonic'  #or 'superharmonic' or 'blend'
        self.blend_alpha = 0.0  #0 = pure sub, 1 = pure super
        self._switch_count = 0

    @property
    def goal_position(self):
        return self.sub.goal_position

    @property
    def obstacles(self):
        return self.sub.obstacles

    @property
    def danger_distance(self):
        return self.sub.danger_distance

    def _update_mode(self, dphi_dt: float):
        """
        Update active mode based on dPhi/dt with hysteresis.

        Parameters:
        - dphi_dt : float - current temporal derivative of the potential
        Returns: none (updates internal state)
        """
        if self.blend_width > 0:
            #smooth blending via sigmoid
            self.blend_alpha = 1.0 / (1.0 + np.exp(
                -(dphi_dt - self.switch_threshold) / self.blend_width))
            self.active_mode = 'blend'
        else:
            #hard switching with hysteresis (Schmitt trigger)
            if self.active_mode == 'subharmonic':
                if dphi_dt >= self.switch_threshold:
                    self.active_mode = 'superharmonic'
                    self._switch_count += 1
            else:
                if dphi_dt < self.switch_threshold - self.hysteresis:
                    self.active_mode = 'subharmonic'
                    self._switch_count += 1

    def compute_potential(self, pos: np.ndarray) -> float:
        """
        Computes potential using active engine or blended.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - potential : float
        """
        dphi_dt_sub = self.sub.compute_dphi_dt(pos)
        self._update_mode(dphi_dt_sub)

        if self.active_mode == 'subharmonic':
            return self.sub.compute_potential(pos)
        elif self.active_mode == 'superharmonic':
            return self.sup.compute_potential(pos)
        else:
            #blend mode
            phi_sub = self.sub.compute_potential(pos)
            phi_sup = self.sup.compute_potential(pos)
            return (1 - self.blend_alpha) * phi_sub + self.blend_alpha * phi_sup

    def compute_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Computes gradient using active engine or blended.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - gradient : np.ndarray
        """
        dphi_dt_sub = self.sub.compute_dphi_dt(pos)
        self._update_mode(dphi_dt_sub)

        if self.active_mode == 'subharmonic':
            return self.sub.compute_gradient(pos)
        elif self.active_mode == 'superharmonic':
            return self.sup.compute_gradient(pos)
        else:
            g_sub = self.sub.compute_gradient(pos)
            g_sup = self.sup.compute_gradient(pos)
            return (1 - self.blend_alpha) * g_sub + self.blend_alpha * g_sup

    def compute_dphi_dt(self, pos: np.ndarray) -> float:
        """
        Computes dPhi/dt using active engine or blended.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - dphi_dt : float
        """
        dphi_sub = self.sub.compute_dphi_dt(pos)
        dphi_sup = self.sup.compute_dphi_dt(pos)

        if self.active_mode == 'subharmonic':
            return dphi_sub
        elif self.active_mode == 'superharmonic':
            return dphi_sup
        else:
            return (1 - self.blend_alpha) * dphi_sub + self.blend_alpha * dphi_sup

    def compute_field_features(self, pos: np.ndarray) -> dict:
        """
        Computes full feature dict with mode information.

        Parameters:
        - pos : np.ndarray - query position
        Returns:
        - features : dict with keys: potential, gradient, grad_mag,
          dphi_dt, active_mode, blend_alpha, switch_count
        """
        dphi_dt_sub = self.sub.compute_dphi_dt(pos)
        self._update_mode(dphi_dt_sub)

        phi = self.compute_potential(pos)
        grad = self.compute_gradient(pos)
        dphi_dt = self.compute_dphi_dt(pos)

        return {
            'potential': phi,
            'gradient': grad,
            'grad_mag': np.linalg.norm(grad),
            'dphi_dt': dphi_dt,
            'active_mode': self.active_mode,
            'blend_alpha': self.blend_alpha,
            'switch_count': self._switch_count,
        }

    def sync_obstacles(self):
        """
        Synchronizes obstacle lists between sub and super engines.
        Call after adding/removing obstacles or updating positions.

        Parameters: none
        Returns: none
        """
        self.sup.obstacles.clear()
        for obs in self.sub.obstacles:
            self.sup.add_obstacle(
                position=obs.position.copy(),
                radius=obs.radius,
                velocity=obs.velocity.copy() if obs.velocity is not None else None,
            )