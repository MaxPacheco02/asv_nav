from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, fabs, atan2
import casadi as ca
import numpy as np


class ASVAcadosModel(AcadosModel):
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


def export_asv_model() -> ASVAcadosModel:
    model_name = "asv_dynamics"

    # =========================================================================
    # Ship parameters — from dynamic_model.h
    # =========================================================================
    m = 4725629.25
    Iz = 5829430000.0
    xg = 0.0

    # Added mass
    X_u_dot = -187765.0
    Y_v_dot = -3780505.0
    Y_r_dot = 0.0
    N_r_dot = -1748950469.0

    # Nonlinear damping
    Xuu = -7057.485120
    Yvv = -3890570.407734
    Yrv = 380816892.435056
    Yvr = -11193515.732379
    Yrr = 16112020985.006985
    Nvv = 138515283.493993
    Nrv = -8148922936.922905
    Nvr = 3491938401.993457
    Nrr = -390901820542.178894

    # Thruster limits for normalization
    Tx_max = 2 * 222224.5896  # ~444 kN
    Ty_max = 2 * 222224.5896  # ~444 kN (both thrusters lateral)
    Tz_max = 222224.5896 * 61.1  # ~13.6M N·m

    # Inertia matrix M (constant) — precomputed numerically
    M_np = np.array(
        [
            [m - X_u_dot, 0.0, 0.0],
            [0.0, m - Y_v_dot, m * xg - Y_r_dot],
            [0.0, m * xg - Y_r_dot, Iz - N_r_dot],
        ]
    )
    M_inv_np = np.linalg.inv(M_np)

    # =========================================================================
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
    # Controls: normalized [Tx, Ty, Tz] in [-1, 1], plus spline dt
    # =========================================================================
    u_Tx = SX.sym("u_Tx")
    u_Ty = SX.sym("u_Ty")
    u_Tz = SX.sym("u_Tz")
    dt_ctrl = SX.sym("dt")

    u_ctrl = vertcat(u_Tx, u_Ty, u_Tz, dt_ctrl)

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
    # Coriolis matrix C(nu) — matching C++ get_decomposed_dyn()
    # =========================================================================
    c0 = m * (xg * yaw + sway)
    c1 = m * surge
    C_RB = SX.zeros(3, 3)
    C_RB[0, 2] = -c0
    C_RB[1, 2] = c1
    C_RB[2, 0] = c0
    C_RB[2, 1] = -c1

    c2 = Y_v_dot * sway + Y_r_dot * yaw
    c3 = X_u_dot * surge
    C_A = SX.zeros(3, 3)
    C_A[0, 2] = c2
    C_A[1, 2] = -c3
    C_A[2, 0] = -c2
    C_A[2, 1] = c3

    C_mat = C_RB + C_A

    # =========================================================================
    # Damping matrix D(nu) — matching C++ (uses |nu|)
    # =========================================================================
    surge_abs = fabs(surge)
    sway_abs = fabs(sway)
    # yaw_abs = fabs(yaw)
    yaw_abs = 0.0  # for large ships

    d0 = -Xuu * surge_abs
    d1 = -Yvv * sway_abs - Yrv * yaw_abs
    d2 = -Yvr * sway_abs - Yrr * yaw_abs
    d3 = -Nvv * sway_abs - Nrv * yaw_abs
    d4 = -Nvr * sway_abs - Nrr * yaw_abs

    D_mat = SX.zeros(3, 3)
    D_mat[0, 0] = d0
    D_mat[1, 1] = d1
    D_mat[1, 2] = d2
    D_mat[2, 1] = d3
    D_mat[2, 2] = d4

    # =========================================================================
    # Thrust: normalized controls → physical forces
    # =========================================================================
    tau_thrust = vertcat(u_Tx * Tx_max, u_Ty * Ty_max, u_Tz * Tz_max)

    # =========================================================================
    # nu_dot = M^{-1} * (tau - C*nu - D*nu)
    # =========================================================================
    nu_vec = vertcat(surge, sway, yaw)
    M_inv_sx = SX(M_inv_np)
    nu_dot = M_inv_sx @ (tau_thrust - C_mat @ nu_vec - D_mat @ nu_vec)

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
        nu_dot[0],
        nu_dot[1],
        nu_dot[2],
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
    t_mod = ca.if_else(ca.logic_and(t_mod < 1e-6, t > 0.1), 1.0, t_mod)
    t_mod = ca.if_else(ca.logic_and(t > spline_ceil, in_last_s), 1.0, t_mod)

    t_la_mod = ca.fmod(t_la, 1.0)
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
    model = ASVAcadosModel()
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
    model.u_labels = ["$u_{Tx}$", "$u_{Ty}$", "$u_{Tz}$", "$\\dot{t}$"]
    model.t_label = "$t$ [s]"

    model.Tx_max = Tx_max
    model.Ty_max = Ty_max
    model.Tz_max = Tz_max

    return model
