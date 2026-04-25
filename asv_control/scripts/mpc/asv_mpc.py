"""
ASV Spline Tracking MPC using ACADOS.

Full 3-DOF Fossen dynamics with normalized Tx / Ty / Tz controls, spline
tracking via alongtrack + crosstrack residuals, and soft ellipsoidal
obstacle avoidance.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from casadi import cos, sin, vertcat
from matplotlib.patches import Polygon, Ellipse

from asv_dynamics import export_asv_model


# =============================================================================
# Configuration
# =============================================================================
TF = 100.0
N_HORIZON = 25
T_SIM = 1000.0

# Avoidance cost: w * (1 / ellipse_val)^AVO_POWER
AVO_POWER = 2.0
D_CLAMP = 1e-3
OBS_N = 3

WARMUP_STEPS = 5

# Ellipse defining ASV footprint (semi-axes, metres)
A_ELLIPSE = 65.0  # longitudinal (bow-stern)
B_ELLIPSE = 20.0  # lateral (beam)
ELLIPSE_OFFSET = 0.0  # shift centre fore/aft if COG isn't midship
R_SAFE_ELLIPSE = 30.0  # extra buffer added to both axes

A_ELL_EFF = A_ELLIPSE + R_SAFE_ELLIPSE
B_ELL_EFF = B_ELLIPSE + R_SAFE_ELLIPSE

# Thruster limits (must match asv_dynamics.py)
TX_MAX = 2 * 222224.5896
TY_MAX = 2 * 222224.5896
TZ_MAX = 222224.5896 * 61.1

# Control bounds
U_FORCE_MIN, U_FORCE_MAX = -1.0, 1.0
DT_MIN, DT_MAX = -0.01, 0.1

# State bounds: surge, sway, yaw-rate
# Physical surge limits (from vehicle specs): [-6.945, 5.9678] m/s
# Tightened to avoid thruster force saturation
SURGE_MIN, SURGE_MAX = -6.945, 5.9678
SWAY_MIN, SWAY_MAX = -0.32, 0.32
YAW_MIN, YAW_MAX = -0.00899, 0.00899

# ASV footprint vertices in body frame (meters), ordered for polygon drawing
ASV_SHAPE = np.array(
    [
        [65.0, 0.0],  # bow
        [45.0, 10.0],  # port shoulder
        [-60.0, 10.0],  # port stern
        [-60.0, -10.0],  # stbd stern
        [45.0, -10.0],  # stbd shoulder
    ]
)

# =============================================================================
# Parameter layout
# =============================================================================
N_SPLINE_PARAMS = 8  # 4 cubic spline coefficients x 2 dimensions
P_SPLINE1 = 0  # Index for first spline
P_SPLINE2 = P_SPLINE1 + N_SPLINE_PARAMS  # (8) Index for second spline
P_WEIGHTS = P_SPLINE2 + N_SPLINE_PARAMS  # (16) Index for weight params
N_WEIGHTS = 9  # Number of weight params
P_AUX = P_WEIGHTS + N_WEIGHTS  # (25) Index where 'additional' params start
P_T_LA = P_AUX + 0  # lookahead value among the spline param
P_IN_LAST_S = P_AUX + 1  # is the ASV in the last spline? param
P_SPLINE_CEIL = P_AUX + 2  # numeric spline ceiling param
P_OBS_VEL = P_AUX + 3  # (28) Index for each obstacle velocity


# =============================================================================
# OCP construction
# =============================================================================
def _avoidance_residual(x_pos, y_pos, psi, obs_x, obs_y, w_avoidance):
    dx = obs_x - x_pos
    dy = obs_y - y_pos
    ox_body = dx * cos(psi) + dy * sin(psi) - ELLIPSE_OFFSET
    oy_body = -dx * sin(psi) + dy * cos(psi)
    ellipse_val = (ox_body / A_ELL_EFF) ** 2 + (oy_body / B_ELL_EFF) ** 2
    return ca.sqrt(w_avoidance) * (1.0 / ca.fmax(ellipse_val, D_CLAMP)) ** (
        AVO_POWER / 2
    )


def _build_residuals(model, terminal: bool):
    """Build stage or terminal residual vector.

    Terminal residuals scale the tracking/state weights by `w_terminal` and
    drop the control-related residuals (no `u` at the terminal node).
    """
    # States
    x_pos, y_pos, psi = model.x[0], model.x[1], model.x[2]
    surge, sway, yaw = model.x[3], model.x[4], model.x[5]
    t_param = model.x[6]

    # Controls
    u_Tx, u_Ty, u_Tz = model.u[0], model.u[1], model.u[2]

    # Weights (from parameter vector)
    w_along = model.p[P_WEIGHTS + 0]
    w_cross = model.p[P_WEIGHTS + 1]
    w_heading = model.p[P_WEIGHTS + 2]
    w_input = model.p[P_WEIGHTS + 3]
    w_surge = model.p[P_WEIGHTS + 4]
    w_sway = model.p[P_WEIGHTS + 5]
    w_yaw = model.p[P_WEIGHTS + 6]
    w_terminal = model.p[P_WEIGHTS + 7]
    w_avoidance = model.p[P_WEIGHTS + 8]

    t_la_param = model.p[P_T_LA]
    last_s_param = model.p[P_IN_LAST_S]
    spline_ceil_param = model.p[P_SPLINE_CEIL]

    # Pick spline 1 vs spline 2 depending on whether we've crossed the ceil
    # and we're not already on the last spline.
    cond_t = ca.logic_and(t_param > spline_ceil_param, last_s_param == 0)
    cond_t_la = ca.logic_and(t_la_param > spline_ceil_param, last_s_param == 0)
    s_x = ca.if_else(cond_t, model.s2_x, model.s_x)
    s_y = ca.if_else(cond_t, model.s2_y, model.s_y)
    psi_ref = ca.if_else(cond_t, model.psi2_ref, model.psi_ref)
    s_la_x = ca.if_else(cond_t_la, model.s2_la_x, model.s_la_x)
    s_la_y = ca.if_else(cond_t_la, model.s2_la_y, model.s_la_y)

    # Terminal residuals inherit w_terminal as an overall multiplier
    scale = ca.sqrt(w_terminal) if terminal else 1.0

    pieces = [
        scale * ca.sqrt(w_cross) * (x_pos - s_x),
        scale * ca.sqrt(w_cross) * (y_pos - s_y),
        scale * ca.sqrt(w_along) * (x_pos - s_la_x),
        scale * ca.sqrt(w_along) * (y_pos - s_la_y),
        scale * ca.sqrt(w_heading) * sin((psi - psi_ref) / 2),
    ]

    if not terminal:
        # Inputs only exist on stage residuals
        pieces += [
            ca.sqrt(w_input) * u_Tx,
            ca.sqrt(w_input) * u_Ty,
            ca.sqrt(w_input) * u_Tz,
        ]

    pieces += [
        scale * ca.sqrt(w_surge) * surge,
        scale * ca.sqrt(w_sway) * sway,
        scale * ca.sqrt(w_yaw) * yaw,
    ]

    # Avoidance cost — active everywhere, grows near obstacles
    for i in range(OBS_N):
        obs_x = model.x[7 + 2 * i]
        obs_y = model.x[8 + 2 * i]
        pieces.append(_avoidance_residual(x_pos, y_pos, psi, obs_x, obs_y, w_avoidance))

    return vertcat(*pieces)


def setup_spline_tracking_ocp(x0, params, Tf, N_horizon) -> AcadosOcpSolver:
    """Build and return an SQP_RTI solver for the spline-tracking OCP."""
    ocp = AcadosOcp()
    model = export_asv_model()
    ocp.model = model

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    # --- Cost: NONLINEAR_LS with sqrt(weight) baked into residuals so that
    #   cost = r^T r = sum_i w_i * err_i^2   with runtime-tunable w_i.
    stage_res = _build_residuals(model, terminal=False)
    terminal_res = _build_residuals(model, terminal=True)

    n_stage = stage_res.shape[0]
    n_terminal = terminal_res.shape[0]
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.cost.W = np.eye(n_stage)
    ocp.cost.W_e = np.eye(n_terminal)
    ocp.cost.W_0 = np.eye(n_stage)
    ocp.cost.yref = np.zeros(n_stage)
    ocp.cost.yref_e = np.zeros(n_terminal)
    ocp.cost.yref_0 = np.zeros(n_stage)

    ocp.model.cost_y_expr = stage_res
    ocp.model.cost_y_expr_e = terminal_res
    ocp.model.cost_y_expr_0 = stage_res

    # --- Constraints ---
    ocp.constraints.x0 = x0

    ocp.constraints.lbu = np.array([U_FORCE_MIN, U_FORCE_MIN, U_FORCE_MIN, DT_MIN])
    ocp.constraints.ubu = np.array([U_FORCE_MAX, U_FORCE_MAX, U_FORCE_MAX, DT_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.lbx = np.array([SURGE_MIN, SWAY_MIN, YAW_MIN])
    ocp.constraints.ubx = np.array([SURGE_MAX, SWAY_MAX, YAW_MAX])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Azimuth thruster feasibility: each thruster's (surge, lateral)
    # command must lie in the unit disk.
    u_Tx, u_Ty, u_Tz = model.u[0], model.u[1], model.u[2]
    h_expr_list = [
        u_Tx**2 + ((u_Tz + 2 * u_Ty) / 2) ** 2,  # Thruster 1
        u_Tx**2 + ((2 * u_Ty - u_Tz) / 2) ** 2,  # Thruster 2
    ]

    for i in range(OBS_N):
        dx = model.x[2 * i + 7] - model.x[0]
        dy = model.x[2 * i + 8] - model.x[1]
        psi = model.x[2]
        ox_body = dx * cos(psi) + dy * sin(psi) - ELLIPSE_OFFSET
        oy_body = -dx * sin(psi) + dy * cos(psi)

        # E >= 1 means outside the ellipse
        ellipse_val = (ox_body / A_ELL_EFF) ** 2 + (oy_body / B_ELL_EFF) ** 2
        h_expr_list.append(ellipse_val)

    # 1. Stage Constraints (Thrusters + Obstacles)
    model.con_h_expr = vertcat(*h_expr_list)
    n_h = 2 + OBS_N

    ocp.constraints.lh = np.array([0.0, 0.0] + [1.0] * OBS_N)
    ocp.constraints.uh = np.array([1.0, 1.0] + [1e6] * OBS_N)

    # Soft constraint indices for stage (obstacles start at index 2)
    ocp.constraints.idxsh = np.arange(2, n_h)

    # Apply heavy L1/L2 penalties to the stage slack variables
    L1_penalty = 1e4
    L2_penalty = 1e5
    TINY_PENALTY = 1e-3  # Prevents the upper-slack singularity

    ocp.cost.zl = np.ones(OBS_N) * L1_penalty
    ocp.cost.zu = np.ones(OBS_N) * TINY_PENALTY
    ocp.cost.Zl = np.ones(OBS_N) * L2_penalty
    ocp.cost.Zu = np.ones(OBS_N) * TINY_PENALTY

    # ==========================================================
    # 2. Terminal Constraints (Obstacles ONLY)
    # ==========================================================
    # Slice the list to skip the 2 thruster constraints
    model.con_h_expr_e = vertcat(*h_expr_list[2:])

    ocp.constraints.lh_e = np.array([1.0] * OBS_N)
    ocp.constraints.uh_e = np.array([1e6] * OBS_N)

    # Soft constraint indices for terminal (obstacles start at index 0 now)
    ocp.constraints.idxsh_e = np.arange(0, OBS_N)

    ocp.cost.zl_e = np.ones(OBS_N) * L1_penalty
    ocp.cost.zu_e = np.ones(OBS_N) * TINY_PENALTY
    ocp.cost.Zl_e = np.ones(OBS_N) * L2_penalty
    ocp.cost.Zu_e = np.ones(OBS_N) * TINY_PENALTY

    ocp.parameter_values = params

    # --- Solver options ---
    opts = ocp.solver_options
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "IRK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = 3
    opts.sim_method_newton_iter = 3
    opts.qp_solver_iter_max = 1000
    opts.nlp_solver_max_iter = 200
    opts.qp_solver_cond_N = N_horizon // 2
    opts.regularize_method = "PROJECT_REDUC_HESS"
    opts.levenberg_marquardt = 1.0
    opts.qp_solver_tol_stat = 1e-6
    opts.qp_solver_tol_eq = 1e-6
    opts.qp_solver_tol_ineq = 1e-6
    opts.qp_solver_tol_comp = 1e-6
    opts.globalization = "MERIT_BACKTRACKING"
    opts.alpha_min = 0.01
    opts.alpha_reduction = 0.5
    opts.nlp_solver_type = "SQP_RTI"

    ocp.code_export_directory = "c_generated_code_asv_ocp"
    return AcadosOcpSolver(ocp, json_file="asv_ocp.json")


def setup_integrator(dt, params) -> AcadosSimSolver:
    """Closed-loop integrator for the simulation."""
    sim = AcadosSim()
    sim.model = export_asv_model()
    sim.solver_options.T = dt
    sim.solver_options.num_steps = 10
    sim.code_export_directory = "c_generated_code_asv_sim"
    sim.parameter_values = params
    return AcadosSimSolver(sim)


# =============================================================================
# Spline helpers (used only for visualization / diagnostic cost)
# =============================================================================
def get_catmull_rom_segment(p0, p1, p2, p3, alpha=1.0, tension=0.2):
    """Return (a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y) for one CR segment."""

    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    t01 = dist(p0, p1) ** alpha
    t12 = dist(p1, p2) ** alpha
    t23 = dist(p2, p3) ** alpha

    s = 1.0 - tension
    m1 = s * (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)))
    m2 = s * (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)))

    a = 2.0 * (p1 - p2) + m1 + m2
    b = -3.0 * (p1 - p2) - 2 * m1 - m2
    c = m1
    d = p1
    return np.array([a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1]])


def evaluate_spline(t, params):
    ax, bx, cx, dx, ay, by, cy, dy = params
    return np.array(
        [
            ax * t**3 + bx * t**2 + cx * t + dx,
            ay * t**3 + by * t**2 + cy * t + dy,
        ]
    )


def evaluate_spline_dot(t, params):
    ax, bx, cx, _, ay, by, cy, _ = params
    return np.array(
        [
            3 * ax * t**2 + 2 * bx * t + cx,
            3 * ay * t**2 + 2 * by * t + cy,
        ]
    )


def _wrap_spline_t(t_val, spline_ceil_val, in_last_s_val):
    """Mirror the acados model's spline-`t` wrapping logic."""
    t_mod = t_val % 1.0
    if t_mod < 1e-6 and t_val > 0.1:
        t_mod = 1.0
    if t_val > spline_ceil_val and in_last_s_val:
        t_mod = 1.0
    return t_mod


def compute_costs(state, spline_params, t_la_val, spline_ceil_val, in_last_s_val):
    """Unweighted cost components, matching the OCP formulation exactly.

    Returns (crosstrack, alongtrack, heading, avoidance).
    """
    x, y, psi = state[0], state[1], state[2]
    t_val = state[6]

    t_mod = _wrap_spline_t(t_val, spline_ceil_val, in_last_s_val)
    t_la_mod = _wrap_spline_t(t_la_val, spline_ceil_val, in_last_s_val)

    s = evaluate_spline(t_mod, spline_params)
    s_la = evaluate_spline(t_la_mod, spline_params)
    s_dot = evaluate_spline_dot(t_mod, spline_params)
    psi_ref = np.arctan2(s_dot[1], s_dot[0])

    crosstrack = (x - s[0]) ** 2 + (y - s[1]) ** 2
    alongtrack = (x - s_la[0]) ** 2 + (y - s_la[1]) ** 2
    heading = np.sin((psi - psi_ref) / 2) ** 2

    # Ellipsoidal avoidance (matches OCP)
    avoidance = 0.0
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    for i in range(OBS_N):
        dx = state[7 + 2 * i] - x
        dy = state[8 + 2 * i] - y
        ox_body = dx * cos_p + dy * sin_p - ELLIPSE_OFFSET
        oy_body = -dx * sin_p + dy * cos_p
        ellipse_val = (ox_body / A_ELL_EFF) ** 2 + (oy_body / B_ELL_EFF) ** 2
        avoidance += (1.0 / max(ellipse_val, D_CLAMP)) ** AVO_POWER

    return crosstrack, alongtrack, heading, avoidance


# =============================================================================
# Simulation scenario
# =============================================================================
@dataclass
class Scenario:
    Tf: float = TF
    N_horizon: int = N_HORIZON
    T_sim: float = T_SIM

    # Closed-loop control period (independent of MPC discretization).
    # Matches the ROS mpc_node timer (50 ms = 20 Hz).
    sim_dt: float = 0.05

    # Render every Nth step. At sim_dt=0.05 (20 Hz), plot_every=5 gives a
    # 4 Hz live plot — fast sim, watchable visualization.
    plot_every: int = 5

    # Initial ASV state (x, y, psi, surge, sway, yaw, t)
    asv0: tuple = (2.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.2)

    # Initial obstacle positions [(x, y), ...]
    obs0: tuple = ((100.0, 40.0), (900.0, 500.0), (200.0, -100.0))

    # Obstacle velocities [(vx, vy), ...]
    obs_vel: tuple = ((2.5, 1.0), (-9.0, -5.0), (-2.0, 4.0))

    # Obstacle world-bounds for reflecting (matches obstacle_publisher.cpp).
    # (x_min, x_max, y_min, y_max)
    obs_bounds: tuple = (-50.0, 1500.0, -500.0, 500.0)

    # Spline control points
    spline_ctrl: tuple = (
        (-10.0, 0.0),
        (-5.0, 0.0),
        (500.0, 200.0),
        (1300.0, -200.0),
    )

    # Cost weights: along, cross, heading, input, surge, sway, yaw, term, avo
    weights: tuple = (0.5, 1.0, 10.0, 0.001, 0.001, 10.0, 0.001, 1.0, 1.0)

    # Aux params: t_la, in_last_s, spline_ceil
    aux: tuple = (1.0, 1.0, 1.0)

    @property
    def mpc_dt(self) -> float:
        """MPC shooting interval (used only inside the solver)."""
        return self.Tf / self.N_horizon

    @property
    def Nsim(self) -> int:
        """Number of closed-loop steps, driven by sim_dt (not MPC dt)."""
        return int(self.T_sim / self.sim_dt)

    def initial_state(self) -> np.ndarray:
        flat_obs = np.array(self.obs0).flatten()
        return np.concatenate([np.array(self.asv0), flat_obs])

    def build_params(self) -> np.ndarray:
        spline_params = get_catmull_rom_segment(
            *[np.array(p) for p in self.spline_ctrl]
        )
        flat_obs_vel = np.array(self.obs_vel).flatten()
        return np.concatenate(
            [
                spline_params,  # spline 1
                spline_params,  # spline 2 (same for single-segment)
                np.array(self.weights),  # cost weights
                np.array(self.aux),  # t_la, in_last_s, spline_ceil
                flat_obs_vel,  # obstacle velocities
            ]
        )


# =============================================================================
# Visualization
# =============================================================================
def _asv_verts(x, y, psi):
    """Rotate and translate the ASV footprint to world coordinates."""
    cp, sp = np.cos(psi), np.sin(psi)
    R = np.array([[cp, -sp], [sp, cp]])
    return (ASV_SHAPE @ R.T) + np.array([x, y])


@dataclass
class SimPlotter:
    """All matplotlib setup + per-step updates in one place."""

    scenario: Scenario
    spline_params: np.ndarray

    fig: plt.Figure | None = None
    _lines: dict = field(init=False, default_factory=dict)
    _artists: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.ion()
        self._setup_trajectory(axes[0, 0])
        self._setup_states(axes[0, 1])
        self._setup_controls(axes[1, 0])
        self._setup_costs(axes[1, 1])
        plt.tight_layout()

    def _setup_trajectory(self, ax):
        t_sp = np.linspace(0, 1, 100)
        pts = np.array([evaluate_spline(t, self.spline_params) for t in t_sp])
        ax.plot(pts[:, 0], pts[:, 1], "b--", linewidth=2, label="Spline reference")

        margin = 50.0
        ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
        ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)

        x0 = self.scenario.initial_state()
        ax.plot(x0[0], x0[1], "go", markersize=10, label="Start")

        self._artists["safety_ellipse"] = Ellipse(
            xy=(0.0, 0.0),
            width=2 * A_ELL_EFF,  # Total length (bow-stern)
            height=2 * B_ELL_EFF,  # Total width (beam)
            angle=0.0,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.add_patch(self._artists["safety_ellipse"])

        self._artists["obs_circles"] = []
        for i in range(OBS_N):
            ox, oy = x0[7 + 2 * i], x0[8 + 2 * i]
            circle = plt.Circle((ox, oy), 3.0, color="red", alpha=0.3)
            ax.add_patch(circle)
            self._artists["obs_circles"].append(circle)
            ax.plot(ox, oy, "rx", markersize=10)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("XY Trajectory")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="datalim")

        (self._lines["traj"],) = ax.plot(
            [], [], "r-", linewidth=1.5, label="ASV trajectory"
        )
        (self._lines["s_t"],) = ax.plot(
            [], [], "b^", markersize=10, zorder=6, label="s(t)"
        )
        self._artists["asv"] = Polygon(
            _asv_verts(x0[0], x0[1], x0[2]),
            closed=True,
            fc="green",
            ec="black",
            lw=1.5,
            zorder=5,
        )
        ax.add_patch(self._artists["asv"])
        ax.legend()

    def _setup_states(self, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State")
        ax.set_title("States vs Time")
        ax.grid(True)
        for key, label in [
            ("psi", "ψ [rad]"),
            ("surge", "surge [m/s]"),
            ("sway", "sway [m/s]"),
            ("yaw", "yaw rate [rad/s]"),
            ("t", "t (spline param)"),
        ]:
            (self._lines[key],) = ax.plot([], [], label=label)
        ax.legend()
        self._artists["ax_states"] = ax

    def _setup_controls(self, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Force / Moment")
        ax.set_title("Control Inputs (physical)")
        ax.grid(True)
        (self._lines["tx"],) = ax.plot([], [], label="$T_x$ [kN]")
        (self._lines["ty"],) = ax.plot([], [], label="$T_y$ [kN]")
        (self._lines["tz"],) = ax.plot([], [], label="$T_z$ [kN·m]")
        ax.legend()
        self._artists["ax_controls"] = ax

    def _setup_costs(self, ax):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cost")
        ax.set_title("Cost Components")
        ax.grid(True)
        ax.set_yscale("log")
        for key, label in [
            ("c_cross", "crosstrack"),
            ("c_along", "alongtrack"),
            ("c_head", "heading"),
            ("c_avoid", "avoidance"),
        ]:
            (self._lines[key],) = ax.plot([], [], label=label)
        ax.legend()
        self._artists["ax_costs"] = ax

    def update(self, step, simX, simU, t_axis, t_axis_u, cost_hist):
        x = simX[step + 1]
        Tx_scale = TX_MAX / 1000.0
        Ty_scale = TY_MAX / 1000.0
        Tz_scale = TZ_MAX / 1000.0

        self._lines["traj"].set_data(simX[: step + 2, 0], simX[: step + 2, 1])
        self._artists["asv"].set_xy(_asv_verts(x[0], x[1], x[2]))
        self._artists["safety_ellipse"].set_center((x[0], x[1]))
        self._artists["safety_ellipse"].set_angle(np.degrees(x[2]))

        for oi in range(OBS_N):
            self._artists["obs_circles"][oi].center = (x[7 + 2 * oi], x[8 + 2 * oi])

        st = evaluate_spline(np.clip(x[6], 0, 1), self.spline_params)
        self._lines["s_t"].set_data([st[0]], [st[1]])

        for key, col in [("psi", 2), ("surge", 3), ("sway", 4), ("yaw", 5), ("t", 6)]:
            self._lines[key].set_data(t_axis, simX[: step + 2, col])
        ax_states = self._artists["ax_states"]
        ax_states.relim()
        ax_states.autoscale_view()

        self._lines["tx"].set_data(t_axis_u, simU[: step + 1, 0] * Tx_scale)
        self._lines["ty"].set_data(t_axis_u, simU[: step + 1, 1] * Ty_scale)
        self._lines["tz"].set_data(t_axis_u, simU[: step + 1, 2] * Tz_scale)
        ax_controls = self._artists["ax_controls"]
        ax_controls.relim()
        ax_controls.autoscale_view()

        for key in ("c_cross", "c_along", "c_head", "c_avoid"):
            self._lines[key].set_data(t_axis_u, cost_hist[key][: step + 1])
        ax_costs = self._artists["ax_costs"]
        ax_costs.relim()
        ax_costs.autoscale_view()

        plt.pause(0.001)


# =============================================================================
# Diagnostics on solver failure
# =============================================================================


def report_solver_failure(step, status, ocp_solver, simX, params, scenario):
    print(f"\n=== SOLVER FAILURE at step {step} (status={status}) ===")

    # Slack violations on nonlinear h constraints
    for j in range(scenario.N_horizon + 1):
        try:
            h_val = ocp_solver.get(j, "sl")
        except (ValueError, RuntimeError):
            continue  # node has no slack variables
        except Exception as e:
            print(f"  Node {j}: unexpected error reading slacks: {e}")
            continue
        if np.any(h_val > 1e-3):
            print(f"  Node {j}: lower slack violation = {h_val}")

    # Obstacle info
    for k in range(OBS_N):
        vx = params[P_OBS_VEL + 2 * k]
        vy = params[P_OBS_VEL + 2 * k + 1]
        ox, oy = simX[step, 7 + 2 * k], simX[step, 8 + 2 * k]
        speed = np.sqrt(vx**2 + vy**2)
        print(
            f"  Obs {k}: pos=({ox:.1f},{oy:.1f}) vel=({vx:.2f},{vy:.2f}) speed={speed:.2f} m/s"
        )

    # Current ASV state
    x, y, psi, surge = simX[step, 0], simX[step, 1], simX[step, 2], simX[step, 3]
    print(f"  ASV: pos=({x:.1f},{y:.1f}) surge={surge:.2f} psi={psi:.3f} rad")
    print(f"  Horizon covers ~{surge * scenario.Tf:.0f} m ahead")


# =============================================================================
# Simulation driver
# =============================================================================


def _pin_x0(ocp_solver, x0):
    """Pin the OCP's initial state to x0 by clamping lbx == ubx at node 0."""
    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)


def _solve_step(step, ocp_solver, simX, t_prep, t_feedback):
    """One RTI step (or full SQP warmup for the first few steps).

    Returns the solver status.
    """
    # Pin initial state
    _pin_x0(ocp_solver, simX[step, :])

    if step < WARMUP_STEPS:
        ocp_solver.options_set("rti_phase", 0)
        status = ocp_solver.solve()
        t_prep[step] = ocp_solver.get_stats("time_tot")
        t_feedback[step] = 0.0
        return status

    # Preparation
    ocp_solver.options_set("rti_phase", 1)
    ocp_solver.solve()
    t_prep[step] = ocp_solver.get_stats("time_tot")

    # Re-pin x0 (state may have moved since prep was queued)
    _pin_x0(ocp_solver, simX[step, :])

    # Feedback
    ocp_solver.options_set("rti_phase", 2)
    status = ocp_solver.solve()
    t_feedback[step] = ocp_solver.get_stats("time_tot")
    return status


def bounce_obstacles(state, params, bounds):
    """Reflect obstacle positions off world bounds and flip velocities.

    Mirrors obstacle_publisher.cpp's reflect-on-wall logic. Mutates both the
    obstacle rows of `state` (so the plant sees the bounce) and the obstacle
    velocities inside `params` (so the MPC's prediction uses the new heading).
    """
    x_min, x_max, y_min, y_max = bounds
    for i in range(OBS_N):
        ox_idx = 7 + 2 * i
        oy_idx = 8 + 2 * i
        vx_idx = P_OBS_VEL + 2 * i
        vy_idx = P_OBS_VEL + 2 * i + 1

        ox, oy = state[ox_idx], state[oy_idx]
        vx, vy = params[vx_idx], params[vy_idx]

        if ox < x_min:
            state[ox_idx] = x_min
            vx = abs(vx)
        elif ox > x_max:
            state[ox_idx] = x_max
            vx = -abs(vx)

        if oy < y_min:
            state[oy_idx] = y_min
            vy = abs(vy)
        elif oy > y_max:
            state[oy_idx] = y_max
            vy = -abs(vy)

        params[vx_idx] = vx
        params[vy_idx] = vy


def run_simulation(scenario: Scenario):
    x0 = scenario.initial_state()
    params = scenario.build_params()
    spline_params = params[P_SPLINE1 : P_SPLINE1 + N_SPLINE_PARAMS]

    print(f"Spline parameters: {spline_params}")
    print("Setting up OCP solver...")
    ocp_solver = setup_spline_tracking_ocp(x0, params, scenario.Tf, scenario.N_horizon)
    # Plant integrator steps at the closed-loop control period (sim_dt),
    # independently of the MPC's internal shooting interval (Tf/N_horizon).
    integrator = setup_integrator(scenario.sim_dt, params)

    Nsim = scenario.Nsim
    nx = x0.shape[0]
    nu = 4  # 3 thruster commands (Tx, Ty, Tz) + 1 slack
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    simX[0, :] = x0

    t_prep = np.zeros(Nsim)
    t_feedback = np.zeros(Nsim)
    cost_hist = {k: np.zeros(Nsim) for k in ("c_cross", "c_along", "c_head", "c_avoid")}

    plotter = SimPlotter(scenario, spline_params)

    print(f"\nRunning closed-loop simulation for {scenario.T_sim}s...")
    print(f"Number of steps: {Nsim}")

    w_along = params[P_WEIGHTS + 0]
    w_cross = params[P_WEIGHTS + 1]
    w_head = params[P_WEIGHTS + 2]
    w_avoid = params[P_WEIGHTS + 8]
    t_la = params[P_T_LA]
    in_last_s = params[P_IN_LAST_S]
    spline_ceil = params[P_SPLINE_CEIL]

    for step in range(Nsim):
        # Params may change at runtime — push them to every stage
        for j in range(scenario.N_horizon + 1):
            ocp_solver.set(j, "p", params)

        status = _solve_step(step, ocp_solver, simX, t_prep, t_feedback)
        if status not in (0, 2):
            report_solver_failure(step, status, ocp_solver, simX, params, scenario)

        simU[step, :] = ocp_solver.get(0, "u")
        simX[step + 1, :] = integrator.simulate(
            x=simX[step, :], u=simU[step, :], p=params
        )
        # Wrap psi to [-pi, pi]
        simX[step + 1, 2] = (simX[step + 1, 2] + np.pi) % (2 * np.pi) - np.pi

        # Reflect obstacles off world bounds. Also mutates params so the MPC's
        # constant-velocity prediction on the next solve uses the new heading.
        bounce_obstacles(simX[step + 1, :], params, scenario.obs_bounds)

        # Cost diagnostics (weighted)
        ct, at, he, av = compute_costs(
            simX[step + 1, :], spline_params, t_la, spline_ceil, in_last_s
        )
        cost_hist["c_cross"][step] = w_cross * ct
        cost_hist["c_along"][step] = w_along * at
        cost_hist["c_head"][step] = w_head * he
        cost_hist["c_avoid"][step] = w_avoid * av

        # Plot update (time axis in real simulation seconds)
        if (step % scenario.plot_every == 0) or (step == Nsim - 1):
            t_axis = np.arange(step + 2) * scenario.sim_dt
            t_axis_u = np.arange(step + 1) * scenario.sim_dt
            plotter.update(step, simX, simU, t_axis, t_axis_u, cost_hist)

        if (step + 1) % 50 == 0 or step == 0:
            x = simX[step, :]
            sim_t = (step + 1) * scenario.sim_dt
            print(
                f"t={sim_t:6.2f}s ({step + 1}/{Nsim}): spline_t={x[6]:.3f}, "
                f"pos=({x[0]:.2f},{x[1]:.2f}), "
                f"surge={x[3]:.3f}, sway={x[4]:.3f}, "
                f"prep={t_prep[step] * 1000:.2f}ms, "
                f"feedback={t_feedback[step] * 1000:.2f}ms"
            )

    print("\n=== Simulation Complete ===")
    t_prep_ms = t_prep * 1000.0
    t_fb_ms = t_feedback * 1000.0
    print(
        f"Preparation [ms]: min={t_prep_ms.min():.3f}, "
        f"median={np.median(t_prep_ms):.3f}, max={t_prep_ms.max():.3f}"
    )
    print(
        f"Feedback    [ms]: min={t_fb_ms.min():.3f}, "
        f"median={np.median(t_fb_ms):.3f}, max={t_fb_ms.max():.3f}"
    )

    plt.ioff()
    plt.show(block=True)


def main():
    # CLI: `python asv_mpc.py [simulate]`  — simulate defaults to True.
    simulate = True
    if len(sys.argv) > 1:
        simulate = sys.argv[1].lower() not in ("false", "0", "no")

    scenario = Scenario()
    if simulate:
        run_simulation(scenario)
    else:
        # Just build the solver (useful for codegen without simulating).
        x0 = scenario.initial_state()
        params = scenario.build_params()
        setup_spline_tracking_ocp(x0, params, scenario.Tf, scenario.N_horizon)


if __name__ == "__main__":
    main()
