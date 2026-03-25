"""
ASV Spline Tracking MPC using ACADOS
Full 3-DOF Fossen dynamics with normalized Tx/Tz controls
"""

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from asv_dynamics import export_asv_model
import numpy as np
import scipy.linalg
from casadi import vertcat, sin, cos, SX, atan2
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def setup_spline_tracking_ocp(x0, params, Tf, N_horizon, algorithm="RTI"):
    ocp = AcadosOcp()

    model = export_asv_model()
    ocp.model = model

    w_along = model.p[16]
    w_cross = model.p[17]
    w_heading = model.p[18]
    w_input = model.p[19]
    w_surge = model.p[20]
    w_sway = model.p[21]
    w_yaw = model.p[22]
    w_terminal = model.p[23]
    w_avoidance = model.p[24]

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # States: [x, y, psi, surge, sway, yaw, t, obs...]
    x_pos = model.x[0]
    y_pos = model.x[1]
    psi = model.x[2]
    surge = model.x[3]
    sway = model.x[4]
    yaw = model.x[5]
    t_param = model.x[6]

    obs_n = model.obs_n
    obs = []
    for i in range(obs_n):
        obs.append(model.x[7 + 2 * i])
        obs.append(model.x[8 + 2 * i])

    t_la_param = model.p[25]
    last_s_param = model.p[26]
    spline_ceil_param = model.p[27]

    # Controls: [u_Tx, u_Ty, u_Tz, dt, slack_u] — normalized [-1,1]
    u_Tx = model.u[0]
    u_Ty = model.u[1]
    u_Tz = model.u[2]
    dt = model.u[3]
    slack_u = model.u[4]

    # Spline evaluation
    condition_t = ca.logic_and(t_param > spline_ceil_param, last_s_param == 0)
    condition_t_la = ca.logic_and(t_la_param > spline_ceil_param, last_s_param == 0)
    s_x = ca.if_else(condition_t, model.s2_x, model.s_x)
    s_y = ca.if_else(condition_t, model.s2_y, model.s_y)
    psi_ref = ca.if_else(condition_t, model.psi2_ref, model.psi_ref)
    s_x_dot = ca.if_else(condition_t, model.s2_x_dot, model.s_x_dot)
    s_y_dot = ca.if_else(condition_t, model.s2_y_dot, model.s_y_dot)

    s_la_x = ca.if_else(condition_t_la, model.s2_la_x, model.s_la_x)
    s_la_y = ca.if_else(condition_t_la, model.s2_la_y, model.s_la_y)

    # --- COST ---
    crosstrack_error = (x_pos - s_x) ** 2 + (y_pos - s_y) ** 2
    alongtrack_error = (x_pos - s_la_x) ** 2 + (y_pos - s_la_y) ** 2
    heading_error = sin((psi - psi_ref) / 2) ** 2

    # Input cost: already normalized, so u_Tx² + u_Ty² + u_Tz² ∈ [0, 3]
    input_cost = u_Tx**2 + u_Ty**2 + u_Tz**2 + dt**2
    surge_cost = surge**2
    sway_cost = sway**2
    yaw_cost = yaw**2

    avoidance_list = [
        [1.2, 0.0],
        [0.35, 0.0],
        [0.45, -0.25],
        [0.45, 0.25],
        [-0.35, -0.3],
        [-0.35, 0.3],
        [-0.35, 0.0],
    ]
    avoidance_cost = 0.0
    for i in range(int(obs_n)):
        for avo in avoidance_list:
            x_virt = x_pos + avo[0] * cos(psi) - avo[1] * sin(psi)
            y_virt = y_pos + avo[0] * sin(psi) + avo[1] * cos(psi)
            avoidance_cost += 1.0 / (
                (
                    np.sqrt((obs[i * 2] - x_virt) ** 2 + (obs[i * 2 + 1] - y_virt) ** 2)
                    / 3.0
                )
                ** 1.5
            )

    stage_cost = (
        w_along * alongtrack_error
        + w_cross * crosstrack_error
        + w_heading * heading_error
        + w_input * input_cost
        + w_surge * surge_cost
        + w_sway * sway_cost
        + w_yaw * yaw_cost
        + w_avoidance * avoidance_cost
    )
    ocp.model.cost_expr_ext_cost = stage_cost

    terminal_cost = w_terminal * (
        w_along * alongtrack_error
        + w_cross * crosstrack_error
        + w_heading * heading_error
        + w_surge * surge_cost
        + w_sway * sway_cost
        + w_yaw * yaw_cost
        + w_avoidance * avoidance_cost
    )
    ocp.model.cost_expr_ext_cost_e = terminal_cost

    # --- CONSTRAINTS ---
    ocp.constraints.x0 = x0

    # Controls: normalized [-1,1] for forces, plus dt and slack
    dt_max = 0.01
    dt_min = -0.001
    slack_u_max = 1.0
    slack_u_min = 0.0

    ocp.constraints.lbu = np.array([-1.0, -1.0, -1.0, dt_min, slack_u_min])
    ocp.constraints.ubu = np.array([1.0, 1.0, 1.0, dt_max, slack_u_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # State bounds: surge, sway, yaw
    u_min, u_max = -0.1, 6.0
    v_min, v_max = -1.0, 1.0
    r_min, r_max = -0.1, 0.1

    ocp.constraints.lbx = np.array([u_min, v_min, r_min])
    ocp.constraints.ubx = np.array([u_max, v_max, r_max])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.model.con_h_expr = vertcat(
        u_Tx**2 + ((u_Tz + 2 * u_Ty) / 2) ** 2,  # thruster 0
        u_Tx**2 + ((2 * u_Ty - u_Tz) / 2) ** 2,  # thruster 1
    )
    ocp.constraints.lh = np.array([0.0, 0.0])
    ocp.constraints.uh = np.array([1.0, 1.0])

    ocp.parameter_values = params

    # --- SOLVER OPTIONS ---
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.sim_method_num_steps = 20  # substeps for stiff dynamics

    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.sim_method_num_stages = 4  # Standard Gauss-Legendre stages
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.sim_method_newton_iter = (
        3  # Newton iterations for the implicit solve
    )

    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_cond_N = N_horizon // 2

    ocp.solver_options.regularize_method = "PROJECT_REDUC_HESS"
    ocp.solver_options.levenberg_marquardt = 1.0

    ocp.solver_options.qp_solver_tol_stat = 1e-6
    ocp.solver_options.qp_solver_tol_eq = 1e-6
    ocp.solver_options.qp_solver_tol_ineq = 1e-6
    ocp.solver_options.qp_solver_tol_comp = 1e-6

    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.alpha_min = 0.01
    ocp.solver_options.alpha_reduction = 0.7

    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    ocp.code_export_directory = "c_generated_code_asv_ocp"

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="asv_ocp.json")
    return acados_ocp_solver


def setup_integrator(dt, params):
    sim = AcadosSim()
    sim.model = export_asv_model()

    sim.solver_options.T = dt
    sim.solver_options.num_steps = 10  # match OCP substeps
    sim.code_export_directory = "c_generated_code_asv_sim"
    sim.parameter_values = params

    return AcadosSimSolver(sim)


def get_catmull_rom_segment(p0, p1, p2, p3, alpha=1.0, tension=0.2):
    def distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    t01 = distance(p0, p1) ** alpha
    t12 = distance(p1, p2) ** alpha
    t23 = distance(p2, p3) ** alpha

    m1 = (1.0 - tension) * (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)))
    m2 = (1.0 - tension) * (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)))

    a = 2.0 * (p1 - p2) + m1 + m2
    b = -3.0 * (p1 - p2) - m1 - m1 - m2
    c = m1
    d = p1

    return np.array([a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1]])


def get_triangle_verts(x, y, psi, L=20.0):
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    bow = [x + L * cos_p, y + L * sin_p]
    port = [
        x - 0.4 * L * cos_p + 0.3 * L * sin_p,
        y - 0.4 * L * sin_p - 0.3 * L * cos_p,
    ]
    stbd = [
        x - 0.4 * L * cos_p - 0.3 * L * sin_p,
        y - 0.4 * L * sin_p + 0.3 * L * cos_p,
    ]
    return [bow, port, stbd]


def evaluate_spline(t, params):
    a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y = params
    x = a_x * t**3 + b_x * t**2 + c_x * t + d_x
    y = a_y * t**3 + b_y * t**2 + c_y * t + d_y
    return np.array([x, y])


def compute_costs(state, spline_params, t_la_val, spline_ceil_val, in_last_s_val):
    x, y, psi = state[0], state[1], state[2]
    t_val = state[6]

    t_mod = t_val % 1.0
    if t_mod < 1e-6 and t_val > 0.1:
        t_mod = 1.0
    if t_val > spline_ceil_val and in_last_s_val:
        t_mod = 1.0
    s = evaluate_spline(t_mod, spline_params)

    t_la_mod = t_la_val % 1.0
    if t_la_mod < 1e-6 and t_la_val > 0.1:
        t_la_mod = 1.0
    if t_la_val > spline_ceil_val and in_last_s_val:
        t_la_mod = 1.0
    s_la = evaluate_spline(t_la_mod, spline_params)

    a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y = spline_params
    sx_dot = 3 * a_x * t_mod**2 + 2 * b_x * t_mod + c_x
    sy_dot = 3 * a_y * t_mod**2 + 2 * b_y * t_mod + c_y
    psi_ref = np.arctan2(sy_dot, sx_dot)

    crosstrack = (x - s[0]) ** 2 + (y - s[1]) ** 2
    alongtrack = (x - s_la[0]) ** 2 + (y - s_la[1]) ** 2
    heading = np.sin((psi - psi_ref) / 2) ** 2

    return crosstrack, alongtrack, heading


def main(algorithm="RTI", simulate=True):
    Tf = 50.0
    N_horizon = 50
    dt = Tf / N_horizon

    # State: [x, y, psi, surge, sway, yaw, t, obs...]
    x0 = np.array(
        [
            2.0,  # x
            -4.0,  # y
            0.0,  # psi
            0.0,  # surge
            0.0,  # sway
            0.0,  # yaw
            0.2,  # t
            100.0,
            100.0,  # obs 0 (far away)
            100.0,
            100.0,  # obs 1
            100.0,
            100.0,  # obs 2
        ]
    )

    T_sim = 1000.0
    Nsim = int(T_sim / dt)
    nx = x0.shape[0]
    nu = 5  # u_Tx, u_Ty, u_Tz, dt, slack_u
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    simX[0, :] = x0
    t_preparation = np.zeros(Nsim)
    t_feedback = np.zeros(Nsim)

    # Spline
    p0 = np.array([-10.0, 0.0])
    p1 = np.array([-5.0, 0.0])
    p2 = np.array([500.0, 200.0])
    p3 = np.array([1300.0, -200.0])

    spline_params = get_catmull_rom_segment(p0, p1, p2, p3)
    print(f"Spline parameters: {spline_params}")

    Tx_max = 2 * 222224.5896
    Ty_max = 2 * 222224.5896
    Tz_max = 222224.5896 * 61.1

    w_params = np.array(
        [
            0.001,  # w_along
            5.0,  # w_cross
            50.0,  # w_heading
            0.01,  # w_input
            0.10,  # w_surge
            100.0,  # w_sway
            0.0,  # w_yaw
            100.0,  # w_terminal
            0.0,  # w_avoidance
        ]
    )
    add_params = np.array([1.0, 1.0, 1.0])
    ov_params = np.zeros(6)

    params = np.concatenate(
        (spline_params, spline_params, w_params, add_params, ov_params)
    )

    print("Setting up OCP solver...")
    ocp_solver = setup_spline_tracking_ocp(x0, params, Tf, N_horizon, algorithm)

    if simulate:
        integrator = setup_integrator(dt, params)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.ion()

        ax_traj = axes[0, 0]
        t_spline = np.linspace(0, 1, 100)
        spline_points = np.array([evaluate_spline(t, spline_params) for t in t_spline])
        ax_traj.plot(
            spline_points[:, 0],
            spline_points[:, 1],
            "b--",
            linewidth=2,
            label="Spline reference",
        )
        ax_traj.plot(x0[0], x0[1], "go", markersize=10, label="Start")
        ax_traj.set_xlabel("X [m]")
        ax_traj.set_ylabel("Y [m]")
        ax_traj.set_title(f"XY Trajectory ({algorithm})")
        ax_traj.legend()
        ax_traj.grid(True)
        ax_traj.axis("equal")

        (line_traj,) = ax_traj.plot([], [], "r-", linewidth=1.5, label="ASV trajectory")
        (point_st,) = ax_traj.plot([], [], "b^", markersize=10, zorder=6, label="s(t)")
        asv_patch = Polygon(
            get_triangle_verts(x0[0], x0[1], x0[2]),
            closed=True,
            fc="green",
            ec="black",
            lw=1.5,
            zorder=5,
        )
        ax_traj.add_patch(asv_patch)

        ax_states = axes[0, 1]
        ax_states.set_xlabel("Time [s]")
        ax_states.set_ylabel("State")
        ax_states.set_title("States vs Time")
        ax_states.grid(True)
        (line_psi,) = ax_states.plot([], [], label="ψ [rad]")
        (line_surge,) = ax_states.plot([], [], label="surge [m/s]")
        (line_sway,) = ax_states.plot([], [], label="sway [m/s]")
        (line_yaw,) = ax_states.plot([], [], label="yaw rate [rad/s]")
        (line_t,) = ax_states.plot([], [], label="t (spline param)")
        ax_states.legend()

        ax_controls = axes[1, 0]
        ax_controls.set_xlabel("Time [s]")
        ax_controls.set_ylabel("Force / Moment")
        ax_controls.set_title("Control Inputs (physical)")
        ax_controls.grid(True)
        (line_tx,) = ax_controls.plot([], [], label="$T_x$ [kN]")
        (line_ty,) = ax_controls.plot([], [], label="$T_y$ [kN]")
        (line_tz,) = ax_controls.plot([], [], label="$T_z$ [kN·m]")
        ax_controls.legend()

        ax_costs = axes[1, 1]
        ax_costs.set_xlabel("Time [s]")
        ax_costs.set_ylabel("Cost")
        ax_costs.set_title("Cost Components")
        ax_costs.grid(True)
        ax_costs.set_yscale("log")
        (line_cross,) = ax_costs.plot([], [], label="crosstrack")
        (line_along,) = ax_costs.plot([], [], label="alongtrack")
        (line_head,) = ax_costs.plot([], [], label="heading")
        ax_costs.legend()

        cost_cross = np.zeros(Nsim)
        cost_along = np.zeros(Nsim)
        cost_head = np.zeros(Nsim)

        plt.tight_layout()

        print(f"\nRunning closed-loop simulation for {T_sim}s with {algorithm}...")
        print(f"Number of steps: {Nsim}")

        for i in range(Nsim):
            for j in range(N_horizon + 1):
                ocp_solver.set(j, "p", params)

            if i < 5:
                ocp_solver.set(0, "lbx", simX[i, :])
                ocp_solver.set(0, "ubx", simX[i, :])
                ocp_solver.options_set("rti_phase", 0)
                status = ocp_solver.solve()
                t_preparation[i] = ocp_solver.get_stats("time_tot")
                t_feedback[i] = 0.0
            else:
                ocp_solver.options_set("rti_phase", 1)
                status = ocp_solver.solve()
                t_preparation[i] = ocp_solver.get_stats("time_tot")

                ocp_solver.set(0, "lbx", simX[i, :])
                ocp_solver.set(0, "ubx", simX[i, :])

                ocp_solver.options_set("rti_phase", 2)
                status = ocp_solver.solve()
                t_feedback[i] = ocp_solver.get_stats("time_tot")

            if status not in [0, 2, 5]:
                print(f"Warning: Solver returned status {status} at step {i}")

            simU[i, :] = ocp_solver.get(0, "u")
            simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :], p=params)

            # Update plots
            time = np.arange(i + 2) * dt
            time_u = np.arange(i + 1) * dt

            line_traj.set_data(simX[: i + 2, 0], simX[: i + 2, 1])
            asv_patch.set_xy(
                get_triangle_verts(simX[i + 1, 0], simX[i + 1, 1], simX[i + 1, 2])
            )
            st_pos = evaluate_spline(np.clip(simX[i + 1, 6], 0, 1), spline_params)
            point_st.set_data([st_pos[0]], [st_pos[1]])

            line_psi.set_data(time, simX[: i + 2, 2])
            line_surge.set_data(time, simX[: i + 2, 3])
            line_sway.set_data(time, simX[: i + 2, 4])
            line_yaw.set_data(time, simX[: i + 2, 5])
            line_t.set_data(time, simX[: i + 2, 6])
            ax_states.relim()
            ax_states.autoscale_view()

            # Plot physical forces (denormalize)
            line_tx.set_data(time_u, simU[: i + 1, 0] * Tx_max / 1000)
            line_ty.set_data(time_u, simU[: i + 1, 1] * Ty_max / 1000)
            line_tz.set_data(time_u, simU[: i + 1, 2] * Tz_max / 1000)
            ax_controls.relim()
            ax_controls.autoscale_view()

            ct, at, he = compute_costs(
                simX[i + 1, :], spline_params, params[25], params[27], params[26]
            )
            cost_cross[i] = ct
            cost_along[i] = at
            cost_head[i] = he

            line_cross.set_data(time_u, cost_cross[: i + 1])
            line_along.set_data(time_u, cost_along[: i + 1])
            line_head.set_data(time_u, cost_head[: i + 1])
            ax_costs.relim()
            ax_costs.autoscale_view()

            plt.pause(0.001)

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"Step {i + 1}/{Nsim}: t={simX[i, 6]:.3f}, "
                    f"pos=({simX[i, 0]:.2f}, {simX[i, 1]:.2f}), "
                    f"surge={simX[i, 3]:.3f}, sway={simX[i, 4]:.3f}, "
                    f"prep={t_preparation[i] * 1000:.2f}ms, feedback={t_feedback[i] * 1000:.2f}ms"
                )

        print(f"\n=== Simulation Complete ({algorithm}) ===")
        t_preparation *= 1000
        t_feedback *= 1000
        print(
            f"Preparation [ms]: min={np.min(t_preparation):.3f}, median={np.median(t_preparation):.3f}, max={np.max(t_preparation):.3f}"
        )
        print(
            f"Feedback [ms]: min={np.min(t_feedback):.3f}, median={np.median(t_feedback):.3f}, max={np.max(t_feedback):.3f}"
        )

        plt.ioff()
        plt.show(block=True)


if __name__ == "__main__":
    # main(algorithm="RTI", simulate=True)
    main(algorithm="RTI", simulate=False)
