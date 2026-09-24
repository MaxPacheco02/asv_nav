from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, fabs, atan2
import casadi as ca
import numpy as np


class USVAcadosModel(AcadosModel):
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
    T_max: float
    T_min: float


def export_usv_model() -> USVAcadosModel:
    model_name = "usv_dynamics"

    # =========================================================================
    # Ship parameters — from dynamic_model.h
    # =========================================================================
    m = 30.0
    B = 0.41
    Iz = 4.1
    xg = 0.0

    # Added mass
    X_u_dot = -2.25
    Y_v_dot = -23.13
    Y_r_dot = -1.31
    N_v_dot = -16.41
    N_r_dot = -2.79

    # Nonlinear damping
    Xuu = -70.92
    Yvv = -99.99
    Yvr = -5.49
    Yrv = -5.49
    Yrr = -8.8
    Nvv = -5.49
    Nvr = -8.8
    Nrv = -8.8
    Nrr = -3.49

    # Thruster limits for normalization
    T_max = 36.5 * 2
    T_min = -30.0 * 2

    # Inertia matrix M (constant) — precomputed numerically
    M_np = np.array(
        [
            [m - X_u_dot, 0.0, 0.0],
            [0.0, m - Y_v_dot, m * xg - Y_r_dot],
            [0.0, m * xg - N_v_dot, Iz - N_r_dot],
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

    # Thrust commands live in the state vector so the true control is their
    # rate of change. Appended last so every existing state index is unchanged.
    u_Tport = SX.sym("u_Tport")
    u_Tstbd = SX.sym("u_Tstbd")

    x = vertcat(x_pos, y_pos, psi, surge, sway, yaw, t, *obs_states, u_Tport, u_Tstbd)

    # =========================================================================
    # Controls: port/starboard thrust RATES, plus spline dt
    # =========================================================================
    du_Tport = SX.sym("du_Tport")
    du_Tstbd = SX.sym("du_Tstbd")
    dt_ctrl = SX.sym("dt")

    u_ctrl = vertcat(du_Tport, du_Tstbd, dt_ctrl)

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

    c2 = 2.0 * (Y_v_dot * sway + 0.5 * (Y_r_dot + N_v_dot) * yaw)
    c3 = X_u_dot * m * surge
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
    yaw_abs = fabs(yaw)

    vel = ca.sqrt(surge**2 + sway**2 + 1e-9)

    # --- Linear damping Dl (velocity-dependent), from dynamic_model.cpp ---
    # Surge drag switches regime above 1.2 m/s: below it the drag is a flat
    # linear term, above it the quadratic Xuu term takes over.
    Xu = ca.if_else(surge > 1.2, 64.55, -25.0)
    Xuu_act = ca.if_else(surge > 1.2, Xuu, 0.0)

    k = 0.09 * 0.09 * 1.01
    Yv = (
        0.5
        * (-40000.0 * sway_abs)
        * (
            1.1
            + 0.0045 * (1.01 / 0.09)
            - 0.1 * (0.27 / 0.09)
            + 0.016 * (0.27 / 0.09) ** 2
        )
    )
    Yr = 6.0 * (-np.pi * 1000.0) * vel * k
    Nv = 0.06 * (-np.pi * 1000.0) * vel * k
    Nr = 0.02 * (-np.pi * 1000.0) * vel * k * 1.01

    # --- D = Dl - Dn ---
    D_mat = SX.zeros(3, 3)
    D_mat[0, 0] = -Xu - Xuu_act * surge_abs
    D_mat[1, 1] = -Yv - (Yvv * sway_abs + Yvr * yaw_abs)
    D_mat[1, 2] = -Yr - (Yrv * sway_abs + Yrr * yaw_abs)
    D_mat[2, 1] = -Nv - (Nvv * sway_abs + Nvr * yaw_abs)
    D_mat[2, 2] = -Nr - (Nrv * sway_abs + Nrr * yaw_abs)

    # =========================================================================
    # Thrust, physical forces
    # =========================================================================
    T_SCALE = T_max
    Tp, Ts = u_Tport * T_SCALE, u_Tstbd * T_SCALE
    tau_thrust = vertcat(
        Tp + Ts,  #
        0,  #
        0.5 * B * (Tp - Ts),  #
    )

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
        du_Tport,
        du_Tstbd,
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
        SX.sym("u_Tport_dot"),
        SX.sym("u_Tstbd_dot"),
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
    model = USVAcadosModel()
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
    model.u_labels = [
        "$\\dot{u}_{Tport}$",
        "$\\dot{u}_{Tstbd}$",
        "$\\dot{t}$",
    ]
    model.t_label = "$t$ [s]"

    model.T_max = T_max
    model.T_min = T_min

    return model
