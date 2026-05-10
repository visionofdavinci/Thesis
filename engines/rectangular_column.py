"""
Rectangular Column Obstacle (C2-smoothed repulsion)

Axis-aligned vertical column with configurable height bounds, with
repulsion defined by the distance to the closest surface point of the
true axis-aligned box. z_lo / z_hi set the vertical extent:
use the workspace bounds for a floor-to-ceiling pillar, or any custom
range so the drone can fly over the top of the column.

Mathematical formulation:
For agent x = (ax, ay, az) and column with footprint center (cx, cy),
half-extents (hx, hy), z-extent [z_lo, z_hi]:

    dx = max(|ax - cx| - hx, 0)
    dy = max(|ay - cy| - hy, 0)
    dz = max(|az - cz| - hz, 0)
    r  = sqrt(dx^2 + dy^2 + dz^2)             (true box surface distance)

The gradient is

    grad Phi(x) = (dPhi/dr) * (s / r)

where s is the signed exterior offset vector (s_i = a_i - c_i  -+  h_i
when the agent is past the +/- face on axis i, 0 between the faces) and
||s|| = r. The unit vector s/r is the outward normal at the nearest
surface point and rotates smoothly as we move from face- to corner-
aligned configurations, so the corner regions need no special handling
beyond the standard C1 kink inherent to box distance fields (which is
unavoidable and does not produce trapping at a fixed radius).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RectangularColumn:
    """
    Axis-aligned vertical column obstacle.

    Attributes:
    center_xy: np.ndarray, shape (2,) - Footprint center in the xy plane.
    half_extents_xy : np.ndarray, shape (2,) - Half-width along x and y.
    z_lo, z_hi      : float - Vertical extent. Set to the workspace bounds for a floor-to-ceiling
        column, or to any custom [z_lo, z_hi] range for a partial-height
        column that the drone can fly over. The repulsive potential is a
        proper 3-D box SDF in all cases, so the gradient pushes the agent
        away from *all* faces -- including the top face when z_hi is
        below the workspace ceiling.
    velocity_xy     : Optional[np.ndarray], shape (2,) - Horizontal translation velocity. None means static.
    label           : str - Human-readable identifier for logging.
    """

    center_xy:       np.ndarray
    half_extents_xy: np.ndarray
    z_lo:            float
    z_hi:            float
    velocity_xy:     Optional[np.ndarray] = None
    label:           str = "column"

    #spherical 

    @property
    def position(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], 0.5 * (self.z_lo + self.z_hi)], dtype=float)

    @property
    def radius(self) -> float:
        # conservative scalar for collision-checking 
        return float(np.linalg.norm(self.half_extents_xy))

    @property
    def velocity(self) -> Optional[np.ndarray]:
        if self.velocity_xy is None:
            return None
        return np.array([self.velocity_xy[0], self.velocity_xy[1], 0.0], dtype=float)

    #internal geometry helpers 

    def _box_params(self):
        cx, cy = self.center_xy
        hx, hy = self.half_extents_xy
        cz = 0.5 * (self.z_lo + self.z_hi)
        hz = 0.5 * (self.z_hi - self.z_lo)
        return cx, cy, cz, hx, hy, hz

    def _signed_offsets(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        Per-axis signed exterior offset:
            >0  : agent past the +face along that axis
            <0  : agent past the -face along that axis
            ==0 : agent between the two faces along that axis
        ||signed_offsets|| equals the true box-surface distance whenever
        the agent is exterior to the box.
        """
        ax, ay, az = agent_pos
        cx, cy, cz, hx, hy, hz = self._box_params()

        def offset(a, c, h):
            d = a - c
            if d >  h: return d - h
            if d < -h: return d + h
            return 0.0

        return np.array([offset(ax, cx, hx),
                         offset(ay, cy, hy),
                         offset(az, cz, hz)], dtype=float)

    def _face_depths(self, agent_pos: np.ndarray) -> np.ndarray:
        """
        Per-axis signed face depth: positive when the agent is between
        the two faces along that axis (depth = how far inside the slab
        it lies on that axis), negative when the agent is exterior.
        """
        ax, ay, az = agent_pos
        cx, cy, cz, hx, hy, hz = self._box_params()
        return np.array([hx - abs(ax - cx), hy - abs(ay - cy), hz - abs(az - cz)], dtype=float)

    def surface_distance(self, agent_pos: np.ndarray) -> float:
        """
        Distance to the nearest surface point. Returns 0 on the surface
        and a negative penetration depth strictly inside the box.
        """
        s = self._signed_offsets(agent_pos)
        r = float(np.linalg.norm(s))
        if r > 0.0:
            return r
        return -float(self._face_depths(agent_pos).min())

    # C2-smoothed radial profile 
    #
    # Phi(r) = a_rep * [ u^(-n) - 1 + n(u-1) - (1/2) n(n+1) (u-1)^2 ]
    # dPhi/dr (r) = (a_rep / d0) * [ -n / u^(n+1)  +  n  -  n(n+1)(u-1) ]
    #     with u = r / d0,  for 0 < r < d0; both quantities are 0 for r >= d0.
    #
    # for construction: subtract the 2nd-order Taylor expansion of u^(-n)
    # about u = 1 from u^(-n). Hence Phi(d0) = Phi'(d0) = Phi''(d0) = 0
    # (C2 across the cutoff), while the singularity at r = 0 is
    # preserved.

    @staticmethod
    def _phi_value(r: float, a_rep: float, n_power: float, d0: float) -> float:
        if r >= d0:
            return 0.0
        u = r / d0
        return a_rep * (
            u ** (-n_power)
            - 1.0
            + n_power * (u - 1.0)
            - 0.5 * n_power * (n_power + 1.0) * (u - 1.0) ** 2
        )

    @staticmethod
    def _phi_radial_derivative(r: float, a_rep: float, n_power: float,
                               d0: float) -> float:
        if r >= d0:
            return 0.0
        u = r / d0
        return (a_rep / d0) * (
            -n_power * u ** (-(n_power + 1.0))
            + n_power
            - n_power * (n_power + 1.0) * (u - 1.0)
        )

    #superharmonic-engine

    def compute_potential(self, agent_pos: np.ndarray, a_rep: float, n_power: float, danger_distance: float, epsilon: float = 1e-4) -> float:
        r = max(self.surface_distance(agent_pos), epsilon)
        return self._phi_value(r, a_rep, n_power, danger_distance)

    def compute_gradient(self, agent_pos: np.ndarray, a_rep: float, n_power: float, danger_distance: float, epsilon: float = 1e-4) -> np.ndarray:
        """
        grad Phi at agent_pos due to this column. Points towards the column (so -grad pushes the agent outward). C2-continuous at
        r = danger_distance.
        """
        s = self._signed_offsets(agent_pos)
        r = float(np.linalg.norm(s))

        # outside the influence region: zero potential and zero gradient,
        # both with vanishing first and second derivatives. No boundary kick.
        if r >= danger_distance:
            return np.zeros(3)

        # interior fallback (agent on or inside the box): push out along the shallowest exit axis with magnitude matching the smoothed
        # exterior |dPhi/dr|(eps), capped for integrator stability.
        if r < epsilon:
            depths = self._face_depths(agent_pos)
            axis = int(np.argmin(np.abs(depths)))   # shallowest exit
            cx, cy, cz, _, _, _ = self._box_params()
            center_on_axis = (cx, cy, cz)[axis]
            sign = np.sign(agent_pos[axis] - center_on_axis) or 1.0
            outward = np.zeros(3)
            outward[axis] = sign

            mag = min(
                abs(self._phi_radial_derivative(epsilon, a_rep, n_power,danger_distance)),
                1e5 * a_rep,
            )
            # gradient points inward (-outward) so -grad is outward
            return -mag * outward

        # exterior:  grad Phi = (dPhi/dr) * (s / r)
        outward = s / r                             # unit normal at nearest surface point
        dphi_dr = self._phi_radial_derivative(r, a_rep, n_power,
                                              danger_distance)
        # dphi_dr <= 0 on (0, d0) -> grad points inward, -grad outward.
        return dphi_dr * outward

    def compute_dphi_dt(self, agent_pos: np.ndarray, a_rep: float, n_power: float, danger_distance: float, epsilon: float = 1e-4) -> float:
        """
        Temporal derivative dPhi/dt for a translating column. Zero for
        static columns - important so they do not trigger any
        mode-switching signal driven by dPhi/dt.

        Phi depends on (agent_pos - col_center), so
            dPhi/dt = grad_col Phi . v_col = -grad_agent Phi . v_col.
        """
        if self.velocity_xy is None:
            return 0.0
        if np.linalg.norm(self.velocity_xy) < 1e-12:
            return 0.0

        grad_agent = self.compute_gradient(agent_pos, a_rep, n_power, danger_distance, epsilon)
        v_col = np.array([self.velocity_xy[0], self.velocity_xy[1], 0.0])
        return float(np.dot(-grad_agent, v_col))

    # motion 

    def update_position(self, dt: float, ws_lo: Optional[float] = None, ws_hi: Optional[float] = None) -> None:
        """
        Integrate column center in xy. With workspace bounds, bounce off
        the in-plane walls. Columns translate only in xy -- z extent
        (z_lo / z_hi) is fixed and does not change during motion.
        """
        if self.velocity_xy is None:
            return
        self.center_xy = self.center_xy + self.velocity_xy * dt
        if ws_lo is not None and ws_hi is not None:
            for ax in range(2):
                lo = ws_lo + self.half_extents_xy[ax]
                hi = ws_hi - self.half_extents_xy[ax]
                if self.center_xy[ax] <= lo:
                    self.center_xy[ax] = lo
                    self.velocity_xy[ax] = abs(self.velocity_xy[ax])
                elif self.center_xy[ax] >= hi:
                    self.center_xy[ax] = hi
                    self.velocity_xy[ax] = -abs(self.velocity_xy[ax])