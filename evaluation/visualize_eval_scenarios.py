"""
Evaluation Scenario Visualizer
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
import os

# add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# project imports - support flat or engines/ layout
from evaluation.eval_config import SCENARIO_FAMILIES, ScenarioConfig, ObstacleDef, FIELD_PARAMS

from engines.superharmonic_field_engine import SuperharmonicFieldEngine


# colour scheme
PALETTE = {
    "workspace":        (0.85, 0.85, 0.85, 0.06),
    "start":            "#2ca02c",     # green
    "goal":             "#d62728",     # red
    "approach_axis":    "#888888",
    "sphere_static":    "#1f77b4",     # blue
    "sphere_moving":    "#ff7f0e",     # orange
    "sphere_chaser":    "#8b0000",     # dark red - distinct from "moving"
    "column":           "#7f4f9e",     # purple
    "velocity":         "#ff4500",
    "chaser_orbit":     "#8b0000",
    "jitter_envelope":  "#888888",
    "start_jitter":     "#2ca02c",
    "ground_shadow":    "#666666",
}

#  RQ family display labels
RQ_LABELS = {
    "RQ1_RQ2": "RQ1/RQ2 -- subharmonic (smoothness + convergence)",
    "RQ3":     "RQ3 -- superharmonic (temporal reactivity)",
    "RQ4":     "RQ4 -- PPO escape (3D static traps)",
    "FULL":    "FULL (ablation 4) -- integration stress test",
}

# explicit camera angle - more side-on than the mpl default to make
# vertical structure readable
VIEW_ELEV = 22
VIEW_AZIM = -58


# 3D primitive renderers

def _wireframe_sphere(ax, center, radius, color, alpha, n_u=24, n_v=18):
    """Wireframe sphere - cheap, no surface fill."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.4)


def _surface_sphere(ax, center, radius, color, alpha):
    """Solid translucent sphere - represents the obstacle body."""
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 24)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0,
                    antialiased=True, shade=True)


def _column_box(ax, center_xy, half_extents_xy, z_lo, z_hi, color, alpha):
    """Static column: rectangular prism spanning the workspace height."""
    cx, cy = center_xy
    hx, hy = half_extents_xy
    xs = [cx - hx, cx + hx]
    ys = [cy - hy, cy + hy]
    zs = [z_lo, z_hi]
    faces = [
        [(xs[0], ys[0], zs[0]), (xs[1], ys[0], zs[0]),
         (xs[1], ys[1], zs[0]), (xs[0], ys[1], zs[0])],   # bottom
        [(xs[0], ys[0], zs[1]), (xs[1], ys[0], zs[1]),
         (xs[1], ys[1], zs[1]), (xs[0], ys[1], zs[1])],   # top
        [(xs[0], ys[0], zs[0]), (xs[0], ys[1], zs[0]),
         (xs[0], ys[1], zs[1]), (xs[0], ys[0], zs[1])],   # x-low
        [(xs[1], ys[0], zs[0]), (xs[1], ys[1], zs[0]),
         (xs[1], ys[1], zs[1]), (xs[1], ys[0], zs[1])],   # x-high
        [(xs[0], ys[0], zs[0]), (xs[1], ys[0], zs[0]),
         (xs[1], ys[0], zs[1]), (xs[0], ys[0], zs[1])],   # y-low
        [(xs[0], ys[1], zs[0]), (xs[1], ys[1], zs[0]),
         (xs[1], ys[1], zs[1]), (xs[0], ys[1], zs[1])],   # y-high
    ]
    coll = Poly3DCollection(faces, facecolor=color, edgecolor='black', alpha=alpha, linewidths=0.5)
    ax.add_collection3d(coll)


def _workspace_box(ax, ws_lo, ws_hi):
    """Faint outline of the workspace cube."""
    L, H = ws_lo, ws_hi
    edges = [
        [(L, L, L), (H, L, L)], [(L, L, L), (L, H, L)], [(L, L, L), (L, L, H)],
        [(H, H, H), (L, H, H)], [(H, H, H), (H, L, H)], [(H, H, H), (H, H, L)],
        [(H, L, L), (H, H, L)], [(H, L, L), (H, L, H)],
        [(L, H, L), (H, H, L)], [(L, H, L), (L, H, H)],
        [(L, L, H), (H, L, H)], [(L, L, H), (L, H, H)],
    ]
    for (a, b) in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='black', alpha=0.18, linewidth=0.6)


def _ground_drop(ax, position, ws_lo, color=None, alpha=0.45, shadow_size=18):
    """
    Vertical dotted line from `position` to the floor at z=ws_lo, plus
    a small filled circle on the floor showing the xy projection.
    """
    if color is None:
        color = PALETTE["ground_shadow"]
    if position[2] - ws_lo > 0.05:
        ax.plot([position[0], position[0]],[position[1], position[1]], [ws_lo, position[2]], color=color, alpha=alpha, linestyle=':', linewidth=0.6)
    ax.scatter(position[0], position[1], ws_lo, color=color, alpha=alpha * 0.6, s=shadow_size, marker='o', edgecolors='none', zorder=2)


def _chaser_orbit_ring(ax, center, radius, color, alpha=0.55,
                       n_points=64):
    """
    Dashed circle in the xy plane at the chaser's z, radius slightly
    larger than the chaser sphere itself. Visual cue that this obstacle
    actively pursues the drone (initial velocity is zero so it would
    otherwise look static).
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = radius * 1.7
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.full_like(theta, center[2])
    ax.plot(x, y, z, color=color, linestyle='--', linewidth=1.4, alpha=alpha)


# scenario rendering

def render_scenario_3d(ax, scenario: ScenarioConfig, title: str = None, show_projections: bool = True):
    """Render a single scenario into a given 3D axis."""
    ws_lo, ws_hi = scenario.ws_lo, scenario.ws_hi

    _workspace_box(ax, ws_lo, ws_hi)

    # start + jitter envelope
    start = scenario.start
    ax.scatter(*start, c=PALETTE["start"], s=150, marker='o', edgecolors='black', linewidths=1.0, label='start', zorder=5)
    if scenario.start_jitter > 0:
        _wireframe_sphere(ax, start, scenario.start_jitter, PALETTE["start_jitter"], alpha=0.3)
    if show_projections:
        _ground_drop(ax, start, ws_lo, color=PALETTE["start"], alpha=0.5, shadow_size=40)

    # goal (no jitter applied to goal in current configs)
    goal = scenario.goal
    ax.scatter(*goal, c=PALETTE["goal"], s=200, marker='*', edgecolors='black', linewidths=1.0, label='goal', zorder=5)
    if show_projections:
        _ground_drop(ax, goal, ws_lo, color=PALETTE["goal"], alpha=0.5, shadow_size=50)

    # straight-line approach axis
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], color=PALETTE["approach_axis"], linestyle='--', linewidth=1.2, alpha=0.6, label='straight-line axis')

    # obstacles
    for i, obs in enumerate(scenario.obstacles):
        if obs.kind == "sphere":
            is_moving = obs.velocity is not None and np.linalg.norm(obs.velocity) > 1e-8
            is_chaser = (i == scenario.chaser_idx and scenario.chaser_speed > 0)

            if is_chaser:
                color = PALETTE["sphere_chaser"]
            elif is_moving:
                color = PALETTE["sphere_moving"]
            else:
                color = PALETTE["sphere_static"]

            _surface_sphere(ax, obs.position, obs.radius, color, alpha=0.55)

            # ground projection for the obstacle
            if show_projections:
                _ground_drop(ax, obs.position, ws_lo, color=PALETTE["ground_shadow"], alpha=0.35, shadow_size=int(140 * obs.radius))

            # jitter envelope around obstacle (radius + pos_jitter)
            if scenario.pos_jitter > 0:
                _wireframe_sphere(ax, obs.position, obs.radius + scenario.pos_jitter,PALETTE["jitter_envelope"], alpha=0.25)

            # velocity arrow if moving (not for the chaser - its initial velocity is zero, but it will move in simulation)
            if is_moving and not is_chaser:
                v = np.asarray(obs.velocity, dtype=float)
                vmag = np.linalg.norm(v)
                # length-scale: 1 m of arrow per 0.5 m/s, capped
                scale = 1.0 / max(vmag, 0.001) * min(vmag * 2.0, 1.0)
                ax.quiver(obs.position[0], obs.position[1], obs.position[2],
                          v[0], v[1], v[2],
                          length=scale, color=PALETTE["velocity"],
                          linewidth=2.0, arrow_length_ratio=0.3,
                          normalize=False)

            # chaser orbit ring + annotation
            if is_chaser:
                _chaser_orbit_ring(ax, obs.position, obs.radius,
                                   PALETTE["chaser_orbit"])
                ax.text(obs.position[0], obs.position[1],
                        obs.position[2] + obs.radius + 0.15,
                        f"chaser  v={scenario.chaser_speed:.2f} m/s",
                        color=PALETTE["sphere_chaser"], fontsize=7,
                        ha='center', alpha=0.9)

        elif obs.kind == "column":
            cx, cy = obs.position[0], obs.position[1]
            hx, hy = obs.half_extents_xy
            _column_box(ax, (cx, cy), (hx, hy), ws_lo, ws_hi,
                        PALETTE["column"], alpha=0.30)
            # column jitter visualised as a footprint expansion outline
            if scenario.pos_jitter > 0:
                jx, jy = hx + scenario.pos_jitter, hy + scenario.pos_jitter
                rect = [
                    (cx - jx, cy - jy, ws_lo + 0.01),
                    (cx + jx, cy - jy, ws_lo + 0.01),
                    (cx + jx, cy + jy, ws_lo + 0.01),
                    (cx - jx, cy + jy, ws_lo + 0.01),
                    (cx - jx, cy - jy, ws_lo + 0.01),
                ]
                xs, ys, zs = zip(*rect)
                ax.plot(xs, ys, zs, color=PALETTE["jitter_envelope"],
                        linewidth=0.6, alpha=0.5, linestyle=':')

    # axes formatting
    ax.set_xlim(ws_lo, ws_hi)
    ax.set_ylim(ws_lo, ws_hi)
    ax.set_zlim(ws_lo, ws_hi)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    if title is None:
        title = scenario.name
    ax.set_title(title, fontsize=10)

    # explicit camera - more side-on than the mpl default so vertical
    # structure (which is the whole point of the v5 redesign) is visible
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def render_scenario_field_slice(ax, scenario: ScenarioConfig,
                                z_slice: float = None,
                                grid_n: int = 80):
    """
    Render a 2D superharmonic potential-field slice at constant z.
    The slice z defaults to (start_z + goal_z) / 2.
    Obstacles whose centre lies within +/- 0.5 m of the slice are
    drawn as filled circles; obstacles further away are drawn as
    dashed circles.
    """
    if z_slice is None:
        z_slice = 0.5 * (scenario.start[2] + scenario.goal[2])

    engine = SuperharmonicFieldEngine(
        goal_position=scenario.goal,
        a_att=FIELD_PARAMS["a_att"],
        a_rep=FIELD_PARAMS["a_rep"],
        n_power=FIELD_PARAMS["n_power"],
        danger_distance=FIELD_PARAMS["danger_distance"],
        workspace_lo=scenario.ws_lo,
        workspace_hi=scenario.ws_hi,
        a_wall=FIELD_PARAMS["a_wall"],
        wall_power=FIELD_PARAMS["wall_power"],
        wall_danger=FIELD_PARAMS["wall_danger"],
    )
    for obs in scenario.obstacles:
        if obs.kind == "sphere":
            engine.add_obstacle(obs.position, obs.radius,
                                obs.velocity if obs.velocity is not None else None)

    # Build the grid
    xs = np.linspace(scenario.ws_lo, scenario.ws_hi, grid_n)
    ys = np.linspace(scenario.ws_lo, scenario.ws_hi, grid_n)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    PHI = np.zeros_like(XX)
    GX = np.zeros_like(XX)
    GY = np.zeros_like(XX)
    for ix in range(grid_n):
        for iy in range(grid_n):
            p = np.array([XX[iy, ix], YY[iy, ix], z_slice])
            PHI[iy, ix] = engine.compute_potential(p)
            g = engine.compute_gradient(p)
            GX[iy, ix] = g[0]
            GY[iy, ix] = g[1]

    # Heatmap (log-compressed). PHI can be very peaky so use log1p.
    phi_show = np.log1p(np.maximum(PHI, 0))
    im = ax.imshow(phi_show,
                   extent=(scenario.ws_lo, scenario.ws_hi,
                           scenario.ws_lo, scenario.ws_hi),
                   origin='lower', cmap='viridis', aspect='equal')

    # Sparse gradient quiver - subsample
    step = max(grid_n // 18, 1)
    QX = XX[::step, ::step]
    QY = YY[::step, ::step]
    # plot -gradient (the descent direction)
    QU = -GX[::step, ::step]
    QV = -GY[::step, ::step]
    qmag = np.hypot(QU, QV) + 1e-9
    ax.quiver(QX, QY, QU / qmag, QV / qmag, color='white',
              alpha=0.7, scale=30, width=0.003,
              headwidth=3, headlength=4)

    # obstacle circles in this slice
    for obs in scenario.obstacles:
        if obs.kind == "sphere":
            dz = abs(obs.position[2] - z_slice)
            in_slice = dz < obs.radius + 0.05
            ls = '-' if in_slice else '--'
            color = PALETTE["sphere_moving"] if (
                obs.velocity is not None and np.linalg.norm(obs.velocity) > 1e-8
            ) else PALETTE["sphere_static"]
            circle = plt.Circle(
                (obs.position[0], obs.position[1]), obs.radius,
                fill=in_slice, alpha=0.4 if in_slice else 0.0,
                edgecolor=color, linewidth=1.5, linestyle=ls,
            )
            ax.add_patch(circle)
        elif obs.kind == "column":
            cx, cy = obs.position[0], obs.position[1]
            hx, hy = obs.half_extents_xy
            rect = plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                 fill=True, alpha=0.4,
                                 edgecolor='black',
                                 facecolor=PALETTE["column"], linewidth=1.0)
            ax.add_patch(rect)

    # start, goal markers (project onto slice)
    ax.scatter(scenario.start[0], scenario.start[1], c=PALETTE["start"],
               s=120, marker='o', edgecolors='white', linewidths=1.5, zorder=5)
    ax.scatter(scenario.goal[0], scenario.goal[1], c=PALETTE["goal"],
               s=180, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
    ax.plot([scenario.start[0], scenario.goal[0]],
            [scenario.start[1], scenario.goal[1]],
            color='white', linestyle='--', linewidth=1.0, alpha=0.7)

    ax.set_xlim(scenario.ws_lo, scenario.ws_hi)
    ax.set_ylim(scenario.ws_lo, scenario.ws_hi)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"{scenario.name}  field slice  z = {z_slice:.2f} m",
                 fontsize=10)
    return im


# orchestration

# clearance diagnostic

def _path_clearance(scenario: ScenarioConfig) -> List[Tuple[int, str, float, float]]:
    """
    For each obstacle, compute the perpendicular distance from the
    straight-line start->goal path to the obstacle CENTER, and the
    surface clearance (center distance minus effective radius).

    The SUB engine's Gaussian repulsion peaks at d* = 1/sqrt(2*k_rep)
    ~= 0.71 m and decreases at smaller distances. So if any obstacle
    has surface_clr < ~0.5 m the drone can drift inside the peak-force
    radius and the SUB engine will not generate enough repulsion to
    deflect it. This function flags such cases.
    """
    s, g = scenario.start, scenario.goal
    path_vec = g - s
    L = float(np.linalg.norm(path_vec))
    d_hat = path_vec / L
    out = []
    for i, o in enumerate(scenario.obstacles):
        if o.kind == "column":
            # represent column as a vertical cylinder at (cx, cy, z_mid)
            c = np.array([o.position[0], o.position[1],
                          (s[2] + g[2]) / 2.0])
            r_eff = float(np.hypot(*o.half_extents_xy))
        else:
            c = o.position
            r_eff = o.radius
        v = c - s
        proj = max(0.0, min(L, float(np.dot(v, d_hat))))
        d_center = float(np.linalg.norm(c - (s + proj * d_hat)))
        d_surface = d_center - r_eff
        out.append((i, o.kind, d_center, d_surface))
    return out


def print_clearance_report(all_scenarios: Dict[str, List[ScenarioConfig]],
                           tight_thresh: float = 0.7,
                           danger_thresh: float = 0.5) -> None:
    """
    Print per-scenario diagnostics so the design intent can be
    verified before running 1000 episodes.

    Two flag levels:
      ``!!`` (surface_clr < danger_thresh): drone will likely collide
             with this obstacle even with FULL config -- design bug.
      ``!``  (surface_clr < tight_thresh): tight but navigable.
             Acceptable for moving obstacles where the geometry is
             time-varying. For static obstacles, prefer >= 0.7 m.
    """
    print("\n" + "=" * 78)
    print(" Path-clearance diagnostic")
    print("   straight-line clearance from each obstacle to start->goal axis")
    print(f"   thresholds: tight (<{tight_thresh:.1f} m) = ! ; "
          f"danger (<{danger_thresh:.1f} m) = !!")
    print("=" * 78)
    for fam, scenarios in all_scenarios.items():
        print(f"\n[{fam}]")
        for sc in scenarios:
            wc = (min(sc.ws_hi - sc.goal[0], sc.goal[0] - sc.ws_lo),
                  min(sc.ws_hi - sc.goal[1], sc.goal[1] - sc.ws_lo),
                  min(sc.ws_hi - sc.goal[2], sc.goal[2] - sc.ws_lo))
            corner_flag = ('  !! GOAL IN CORNER (>=2 walls within 0.8 m)'
                           if sum(w < 0.8 for w in wc) >= 2 else '')
            print(f"  {sc.name}{corner_flag}")
            print(f"    goal_wall_clr=({wc[0]:.2f}, {wc[1]:.2f}, {wc[2]:.2f})  "
                  f"pos_jitter={sc.pos_jitter:.2f}")
            rep = _path_clearance(sc)
            n_danger = sum(1 for r in rep if r[3] < danger_thresh)
            n_tight = sum(1 for r in rep if danger_thresh <= r[3] < tight_thresh)
            for i, kind, d_c, d_s in rep:
                if d_s < danger_thresh:
                    flag = '!!'
                elif d_s < tight_thresh:
                    flag = '! '
                else:
                    flag = '  '
                # only print the ones that fail or are tight
                if d_s < tight_thresh:
                    print(f"    {flag} obs[{i}] {kind:7s}  "
                          f"surface_clr={d_s:+.2f}")
            if n_danger == 0 and n_tight == 0:
                print(f"    OK ({len(rep)} obstacles, all >= {tight_thresh:.1f} m)")
    print("=" * 78 + "\n")


def _scenario_summary_lines(scenario: ScenarioConfig) -> List[str]:
    """Two-line caption for grid plots."""
    n_static = sum(1 for o in scenario.obstacles
                   if o.kind == "sphere" and (o.velocity is None
                                              or np.linalg.norm(o.velocity) < 1e-8))
    n_moving = sum(1 for o in scenario.obstacles
                   if o.kind == "sphere" and (o.velocity is not None
                                              and np.linalg.norm(o.velocity) >= 1e-8))
    n_columns = sum(1 for o in scenario.obstacles if o.kind == "column")
    has_chaser = (scenario.chaser_idx >= 0 and scenario.chaser_speed > 0)
    parts = []
    if n_moving:
        parts.append(f"{n_moving} moving")
    if n_static:
        parts.append(f"{n_static} static")
    if n_columns:
        parts.append(f"{n_columns} columns")
    if has_chaser:
        parts.append(f"+chaser@{scenario.chaser_speed:.2f}m/s")
    line1 = ', '.join(parts) if parts else 'no obstacles'
    d = float(np.linalg.norm(scenario.goal - scenario.start))
    dz = float(scenario.goal[2] - scenario.start[2])
    line2 = (f"d(s,g)={d:.2f} m | dz={dz:+.2f} m | "
             f"pos_j={scenario.pos_jitter:.2f} | "
             f"start_j={scenario.start_jitter:.2f}")
    return [line1, line2]


def render_family_individual(family_name: str,
                             scenarios: List[ScenarioConfig],
                             out_dir: str,
                             with_slice: bool = False,
                             dpi: int = 130) -> List[str]:
    """One PNG per scenario in this family. Returns list of saved paths."""
    family_dir = os.path.join(out_dir, family_name)
    os.makedirs(family_dir, exist_ok=True)
    saved = []
    for sc in scenarios:
        if with_slice:
            fig = plt.figure(figsize=(13, 5.6))
            ax3d = fig.add_subplot(1, 2, 1, projection='3d')
            render_scenario_3d(ax3d, sc, title=f"{sc.name}  (3D scene)")
            ax3d.legend(loc='upper left', fontsize=7, framealpha=0.85)
            ax2d = fig.add_subplot(1, 2, 2)
            im = render_scenario_field_slice(ax2d, sc)
            cbar = fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\log(1 + \Phi)$', fontsize=8)
        else:
            fig = plt.figure(figsize=(7.5, 6.5))
            ax3d = fig.add_subplot(1, 1, 1, projection='3d')
            render_scenario_3d(ax3d, sc)
            ax3d.legend(loc='upper left', fontsize=8, framealpha=0.85)
        # caption
        caption = ' | '.join(_scenario_summary_lines(sc))
        fig.suptitle(f"[{family_name}]  {caption}", fontsize=9, y=0.98)
        path = os.path.join(family_dir, f"{sc.name}.png")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
    return saved


def render_grid_summary(all_scenarios: Dict[str, List[ScenarioConfig]],
                        out_dir: str, dpi: int = 130) -> str:
    """One big figure: all scenarios as 3D thumbnails, grouped by family."""
    families = list(all_scenarios.keys())
    max_per_family = max(len(all_scenarios[f]) for f in families)
    n_rows = len(families)
    n_cols = max_per_family

    fig = plt.figure(figsize=(4.5 * n_cols, 4.0 * n_rows))
    for r, fam in enumerate(families):
        scenarios = all_scenarios[fam]
        for c, sc in enumerate(scenarios):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1,
                                 projection='3d')
            # for the grid view we keep ground projections on - they're
            # the cheapest way to communicate vertical structure when
            # subplots are small
            render_scenario_3d(ax, sc, title=sc.name,
                               show_projections=True)
            ax.text2D(0.02, -0.05,
                      ' | '.join(_scenario_summary_lines(sc)),
                      transform=ax.transAxes, fontsize=7,
                      verticalalignment='top', color='#444')
        # add a per-row label on the leftmost subplot
        ax_first = fig.axes[r * n_cols]
        ax_first.text2D(-0.10, 1.10, RQ_LABELS.get(fam, fam),
                        transform=ax_first.transAxes,
                        fontsize=11, fontweight='bold', color='#222')
    fig.suptitle("Evaluation scenarios -- by RQ family", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, "scenarios_grid.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="scenario_visualizations",
                        help="Where to save the PNGs.")
    parser.add_argument("--slice", action="store_true",
                        help="Also render a 2D potential-field slice next to each 3D view.")
    parser.add_argument("--show", action="store_true",
                        help="Display interactively instead of saving.")
    parser.add_argument("--families", nargs="+",
                        default=["RQ1_RQ2", "RQ3", "RQ4", "FULL"],
                        help="Which families to visualise.")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    if args.show:
        # use a real backend for interactive display
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

    os.makedirs(args.out_dir, exist_ok=True)

    all_scenarios: Dict[str, List[ScenarioConfig]] = {}
    for fam in args.families:
        if fam not in SCENARIO_FAMILIES:
            print(f"  [skip] unknown family '{fam}'")
            continue
        all_scenarios[fam] = SCENARIO_FAMILIES[fam]()

    # print the clearance diagnostic before rendering
    print_clearance_report(all_scenarios)

    # individual figures
    print(f"Saving per-scenario figures to {args.out_dir}/<family>/")
    for fam, scenarios in all_scenarios.items():
        paths = render_family_individual(
            fam, scenarios, args.out_dir,
            with_slice=args.slice, dpi=args.dpi,
        )
        for p in paths:
            print(f"  + {p}")

    # summary grid
    grid_path = render_grid_summary(all_scenarios, args.out_dir, dpi=args.dpi)
    print(f"\nSummary grid: {grid_path}")

    if args.show:
        # quick interactive view of the grid
        img = plt.imread(grid_path)
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title("Evaluation scenarios summary grid (close to exit)")
        plt.show()


if __name__ == "__main__":
    main()