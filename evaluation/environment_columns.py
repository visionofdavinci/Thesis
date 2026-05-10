"""
Integration Patch: Add Column Support to eval_environment.py
- how to add the eval column class to analytical fields
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from engines.rectangular_column import RectangularColumn


@dataclass
class EvalColumnObstacle:
    """
    Mirrors EvalObstacle's interface (position, radius, velocity,
    update_position) so the environment's sphere-centric step loop can
    iterate over a mixed list without special cases for motion updates.

    Also holds a RectangularColumn instance for field-engine calls that
    need box-geometry-aware distance / gradient / dphi_dt.
    """
    column: RectangularColumn

    @property
    def position(self) -> np.ndarray:
        return self.column.position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        # assigning position sets the column center in xy
        self.column.center_xy = np.array([value[0], value[1]], dtype=float)

    @property
    def radius(self) -> float:
        return self.column.radius

    @property
    def velocity(self) -> Optional[np.ndarray]:
        return self.column.velocity

    @velocity.setter
    def velocity(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self.column.velocity_xy = None
        else:
            self.column.velocity_xy = np.array([value[0], value[1]], dtype=float)

    def update_position(self, dt: float, ws_lo: Optional[float] = None, ws_hi: Optional[float] = None) -> None:
        self.column.update_position(dt, ws_lo, ws_hi)

    def surface_distance(self, agent_pos: np.ndarray) -> float:
        """Used by the collision check: distance from agent to column surface."""
        return self.column.surface_distance(agent_pos)


def build_obstacle(odef) -> object:
    if odef.kind == "column":
        half = odef.half_extents_xy
        assert half is not None, "column ObstacleDef must have half_extents_xy"
        col = RectangularColumn(
            center_xy=np.array([odef.position[0], odef.position[1]], dtype=float),
            half_extents_xy=np.array([half[0], half[1]], dtype=float),
            z_lo=-3.0,  # will be overridden by set_workspace below
            z_hi=3.0,
            velocity_xy=(np.array([odef.velocity[0], odef.velocity[1]], dtype=float) if odef.velocity is not None else None),
            label="column",
        )
        return EvalColumnObstacle(column=col)
    # for spheres, the environment keeps using its existing EvalObstacle
    # builder (this function should not be called for spheres)
    raise ValueError(f"build_obstacle only handles columns; got kind={odef.kind}")


# helpers to mix columns into the analytical field engines 

def subharmonic_potential_contribution(column: RectangularColumn, agent_pos: np.ndarray, a_rep: float, k_rep: float) -> float:
    """
    Column contribution to the subharmonic potential.
    Uses the closest-surface distance r and the Gaussian form:
        phi = a_rep * exp(-k_rep * r^2)
    """
    r = column.surface_distance(agent_pos)
    if r < 0:
        r = 0.0  # inside: treat as on the surface for the Gaussian
    return float(a_rep * np.exp(-k_rep * r * r))


def subharmonic_gradient_contribution(column: RectangularColumn, agent_pos: np.ndarray, a_rep: float, k_rep: float, epsilon: float = 1e-4) -> np.ndarray:
    """
    Column gradient for subharmonic engine.
    phi = a_rep * exp(-k r^2), so grad = -2 k a_rep exp(-k r^2) * (r * dr/dagent)
    """
    d_comp = column._signed_offsets(agent_pos)
    r = float(np.linalg.norm(d_comp))
    if r < epsilon:
        # inside the column: use same outward push as superharmonic interior
        return column.compute_gradient(agent_pos, a_rep=a_rep * 10.0, n_power=2.0, danger_distance=1.0, epsilon=epsilon)
    # surface-to-agent unit vector
    ax, ay, az = agent_pos
    cx, cy = column.center_xy
    hx, hy = column.half_extents_xy
    cz = 0.5 * (column.z_lo + column.z_hi)
    hz = 0.5 * (column.z_hi - column.z_lo)

    def signed_offset(a, c, h):
        if a - c > h:
            return a - c - h
        if a - c < -h:
            return a - c + h
        return 0.0
    s = np.array([signed_offset(ax, cx, hx), signed_offset(ay, cy, hy), signed_offset(az, cz, hz)])
    dr_dagent = s / r
    coeff = -2.0 * k_rep * a_rep * np.exp(-k_rep * r * r) * r
    return coeff * dr_dagent


def subharmonic_dphi_dt_contribution(column: RectangularColumn, agent_pos: np.ndarray, a_rep: float, k_rep: float) -> float:
    """Column dphi/dt for subharmonic engine."""
    if column.velocity_xy is None:
        return 0.0
    if np.linalg.norm(column.velocity_xy) < 1e-12:
        return 0.0
    grad_agent = subharmonic_gradient_contribution(column, agent_pos, a_rep, k_rep)
    v_col_3d = np.array([column.velocity_xy[0], column.velocity_xy[1], 0.0])
    return float(np.dot(-grad_agent, v_col_3d))

