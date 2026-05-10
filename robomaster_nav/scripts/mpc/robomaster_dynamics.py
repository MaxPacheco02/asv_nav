from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, fabs, atan2
import casadi as ca
import numpy as np


class RoboMasterAcadosModel(AcadosModel):
    s_x: ca.SX
    s_y: ca.SX
    s_la_x: ca.SX
    s_la_y: ca.SX
    psi_ref: ca.SX
    s2_x: ca.SX
    s2_y: ca.SX
    s_x_dot: ca.SX
    s_y_dot: ca.SX
    s2_x_dot: ca.SX
    s2_y_dot: ca.SX
    s2_la_x: ca.SX
    s2_la_y: ca.SX
    psi2_ref: ca.SX
    obs_n: int
    Tx_max: float
    Ty_max: float
    Tz_max: float


def export_robomaster_model() -> RoboMasterAcadosModel:
    model_name = "robomaster_dynamics"

    # =========================================================================
    # RoboMaster s1 parameters
    # =========================================================================
    lix = 0.1
    liy = 0.1
    beta = 1. / (lix + liy)
    
    obs_n = 3

    # =========================================================================
    # States: [x, y, psi, surge, sway, yaw, t, obs...]
    # =========================================================================
    x_pos = SX.sym("x_pos")
    y_pos = SX.sym("y_pos")
    psi = SX.sym("psi")
    surge = SX.sym("surge")
    sway = SX.sym("sway")
    yaw = SX.sym("yaw")
    t = SX.sym("t")

    obs_states = []
    for i in range(obs_n):
        obs_states.append(SX.sym(f"obs_x_{i}"))
        obs_states.append(SX.sym(f"obs_y_{i}"))

    x = vertcat(x_pos, y_pos, psi, surge, sway, yaw, t, *obs_states)

    # =========================================================================
    # Controls: Individual wheel speed, plus spline dt
    # =========================================================================
    v0 = SX.sym("v0")
    v1 = SX.sym("v1")
    v2 = SX.sym("v2")
    v3 = SX.sym("v3")
    dt_ctrl = SX.sym("dt")

    u_ctrl = vertcat(v0, v1, v2, v3, dt_ctrl)

    # =========================================================================
    # Parameters
    # =========================================================================
    a_x = SX.sym("a_x")
    b_x = SX.sym("b_x")
    c_x = SX.sym("c_x")
    d_x = SX.sym("d_x")
    a_y = SX.sym("a_y")
    b_y = SX.sym("b_y")
    c_y = SX.sym("c_y")
    d_y = SX.sym("d_y")
    a2_x = SX.sym("a2_x")
    b2_x = SX.sym("b2_x")
    c2_x = SX.sym("c2_x")
    d2_x = SX.sym("d2_x")
    a2_y = SX.sym("a2_y")
    b2_y = SX.sym("b2_y")
    c2_y = SX.sym("c2_y")
    d2_y = SX.sym("d2_y")

    w_along = SX.sym("w_along")
    w_cross = SX.sym("w_cross")
    w_heading = SX.sym("w_heading")
    w_input = SX.sym("w_input")
    w_surge = SX.sym("w_surge")
    w_sway = SX.sym("w_sway")
    w_yaw = SX.sym("w_yaw")
    w_terminal = SX.sym("w_terminal")
    w_avoidance = SX.sym("w_avoidance")

    t_la = SX.sym("t_la")
    in_last_s = SX.sym("in_last_s")
    spline_ceil = SX.sym("spline_ceil")

    obs_velocities = []
    for i in range(obs_n):
        obs_velocities.append(SX.sym(f"obs_vx_{i}"))
        obs_velocities.append(SX.sym(f"obs_vy_{i}"))

    p = vertcat(
        a_x,
        b_x,
        c_x,
        d_x,
        a_y,
        b_y,
        c_y,
        d_y,
        a2_x,
        b2_x,
        c2_x,
        d2_x,
        a2_y,
        b2_y,
        c2_y,
        d2_y,
        w_along,
        w_cross,
        w_heading,
        w_input,
        w_surge,
        w_sway,
        w_yaw,
        w_terminal,
        w_avoidance,
        t_la,
        in_last_s,
        spline_ceil,
        *obs_velocities,
    )

    # =========================================================================
    # Kinematics
    # =========================================================================
    cos_psi = cos(psi)
    sin_psi = sin(psi)

    # =========================================================================
    # Obstacle dynamics
    # =========================================================================
    obs_dynamics = []
    for i in range(obs_n):
        obs_dynamics.append(obs_velocities[2 * i])
        obs_dynamics.append(obs_velocities[2 * i + 1])

    # =========================================================================
    # Full explicit ODE
    # =========================================================================
    f_expl = vertcat(
        surge * cos_psi - sway * sin_psi,
        surge * sin_psi + sway * cos_psi,
        yaw,
        (v0 + v1 + v2 + v3) / 4.,
        (-v0 + v1 + v2 - v3) / 4.,
        beta * (-v0 + v1 - v2 + v3) / 4.,
        dt_ctrl,
        *obs_dynamics,
    )

    # Implicit form
    x_dot_sym = SX.sym("x_dot")
    y_dot_sym = SX.sym("y_dot")
    psi_dot_sym = SX.sym("psi_dot")
    surge_dot = SX.sym("surge_dot")
    sway_dot = SX.sym("sway_dot")
    yaw_dot = SX.sym("yaw_dot")
    t_dot_sym = SX.sym("t_dot")
    obs_dots = []
    for i in range(obs_n):
        obs_dots.append(SX.sym(f"obs_x_dot_{i}"))
        obs_dots.append(SX.sym(f"obs_y_dot_{i}"))
    xdot = vertcat(
        x_dot_sym,
        y_dot_sym,
        psi_dot_sym,
        surge_dot,
        sway_dot,
        yaw_dot,
        t_dot_sym,
        *obs_dots,
    )

    f_impl = xdot - f_expl

    # =========================================================================
    # Spline evaluation
    # =========================================================================
    t_mod = ca.fmod(t, 1.0)
    t_mod = ca.if_else(t < 0, 0.0, t_mod)
    t_mod = ca.if_else(ca.logic_and(t_mod < 1e-6, t > 0.1), 1.0, t_mod)
    t_mod = ca.if_else(ca.logic_and(t > spline_ceil, in_last_s), 1.0, t_mod)

    t_la_mod = ca.fmod(t_la, 1.0)
    t_la_mod = ca.if_else(t_la < 0, 0.0, t_la_mod)
    t_la_mod = ca.if_else(ca.logic_and(t_la_mod < 1e-6, t_la > 0.1), 1.0, t_la_mod)
    t_la_mod = ca.if_else(ca.logic_and(t_la > spline_ceil, in_last_s), 1.0, t_la_mod)

    s_x = a_x * t_mod**3 + b_x * t_mod**2 + c_x * t_mod + d_x
    s_y = a_y * t_mod**3 + b_y * t_mod**2 + c_y * t_mod + d_y
    s_la_x = a_x * t_la_mod**3 + b_x * t_la_mod**2 + c_x * t_la_mod + d_x
    s_la_y = a_y * t_la_mod**3 + b_y * t_la_mod**2 + c_y * t_la_mod + d_y
    s_x_dot = 3 * a_x * t_mod**2 + 2 * b_x * t_mod + c_x
    s_y_dot = 3 * a_y * t_mod**2 + 2 * b_y * t_mod + c_y
    psi_ref = atan2(s_y_dot, s_x_dot)

    s2_x = a2_x * t_mod**3 + b2_x * t_mod**2 + c2_x * t_mod + d2_x
    s2_y = a2_y * t_mod**3 + b2_y * t_mod**2 + c2_y * t_mod + d2_y
    s2_la_x = a2_x * t_la_mod**3 + b2_x * t_la_mod**2 + c2_x * t_la_mod + d2_x
    s2_la_y = a2_y * t_la_mod**3 + b2_y * t_la_mod**2 + c2_y * t_la_mod + d2_y
    s2_x_dot = 3 * a2_x * t_mod**2 + 2 * b2_x * t_mod + c2_x
    s2_y_dot = 3 * a2_y * t_mod**2 + 2 * b2_y * t_mod + c2_y
    psi2_ref = atan2(s2_y_dot, s2_x_dot)

    # =========================================================================
    # Assemble model
    # =========================================================================
    model = RoboMasterAcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u_ctrl
    model.p = p
    model.name = model_name

    model.s_x = s_x
    model.s_y = s_y
    model.s_la_x = s_la_x
    model.s_la_y = s_la_y
    model.s_x_dot = s_x_dot
    model.s_y_dot = s_y_dot
    model.s2_x_dot = s2_x_dot
    model.s2_y_dot = s2_y_dot
    model.psi_ref = psi_ref
    model.s2_x = s2_x
    model.s2_y = s2_y
    model.s2_la_x = s2_la_x
    model.s2_la_y = s2_la_y
    model.psi2_ref = psi2_ref
    model.obs_n = obs_n

    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        "$\\psi$ [rad]",
        "$u$ [m/s]",
        "$v$ [m/s]",
        "$r$ [rad/s]",
        "$t$",
    ]
    model.u_labels = ["$u_v0$", "$u_v1$", "$u_v2$", "$u_v3$", "$\\dot{t}$"]
    model.t_label = "$t$ [s]"

    model.v_max = 2.5

    return model
