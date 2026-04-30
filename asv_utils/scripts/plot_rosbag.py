"""
plot_mpc_results.py
====================
Publication-quality figure generator for ASV MPC + AITSMC results.
Reads a ROS 2 SQLite3 bag file and produces figures suitable for a paper.

Usage:
    python3 plot_mpc_results.py <path_to_bag_dir>

Dependencies (rosbags 0.11.x):
    pip install "rosbags>=0.11" matplotlib scipy numpy

Output:
    <bag_dir>/figures/  — one PDF + PNG per figure.
"""

import sys
import argparse
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path

# ── rosbags 0.11.x ────────────────────────────────────────────────────────────
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Typestore, get_typestore
    from rosbags.typesys import get_types_from_msg, get_types_from_idl

    _ROSBAGS_NEW = True
except ImportError:
    try:
        # Fallback: older 0.9.x API
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore

        _ROSBAGS_NEW = False
    except ImportError:
        sys.exit("rosbags not found.  Install with:  pip install rosbags")

# =============================================================================
# Custom asv_interfaces message definitions (inlined — no ROS install needed)
# =============================================================================
# Format: (typename, msg_definition_string)
ASV_MSGS = [
    (
        "asv_interfaces/msg/Obstacle",
        "float64 x\nfloat64 y\nfloat64 v_x\nfloat64 v_y\nint8 color\nstring type\nstring uuid\n",
    ),
    ("asv_interfaces/msg/ObstacleList", "asv_interfaces/msg/Obstacle[] obs_list\n"),
    (
        "asv_interfaces/msg/Ref",
        "float64 x\nfloat64 y\nfloat64 psi\nfloat64 u\nfloat64 u_dot\nfloat64 v\nfloat64 v_dot\nfloat64 r\nfloat64 r_dot\n",
    ),
    (
        "asv_interfaces/msg/State",
        "float64 x\nfloat64 y\nfloat64 psi\nfloat64 u\nfloat64 v\nfloat64 r\nfloat64 u_dot\nfloat64 v_dot\nfloat64 r_dot\n",
    ),
    (
        "asv_interfaces/msg/Thrust",
        "float64 force0\nfloat64 force1\nfloat64 ang0\nfloat64 ang1\n",
    ),
    (
        "asv_interfaces/msg/AitsmcDebug",
        "float64 e\nfloat64 e_i\nfloat64 e_i_dot\nfloat64 s\nfloat64 k\nfloat64 u\n",
    ),
]


def _build_typestore():
    """Build typestore with builtins + custom asv_interfaces types."""
    if _ROSBAGS_NEW:
        # rosbags 0.10+ API
        # get_typestore returns a Typestore pre-loaded with a dialect
        from rosbags.typesys import get_typestore, Typestore

        # 0.11 uses string identifiers instead of Stores enum
        try:
            ts = get_typestore(Typestore.ROS2_HUMBLE)
        except (AttributeError, TypeError):
            # Some 0.10/0.11 versions accept a string
            try:
                ts = get_typestore("ros2_humble")
            except Exception:
                ts = get_typestore(None)  # default

        add_dict = {}
        for typename, msgdef in ASV_MSGS:
            try:
                add_dict.update(get_types_from_msg(msgdef, typename))
            except Exception as e:
                print(f"  [warn] {typename}: {e}")
        ts.register(add_dict)
        return ts
    else:
        # rosbags 0.9.x
        from rosbags.typesys import Stores, get_typestore
        from rosbags.typesys import get_types_from_msg

        ts = get_typestore(Stores.ROS2_HUMBLE)
        add_dict = {}
        for typename, msgdef in ASV_MSGS:
            try:
                add_dict.update(get_types_from_msg(msgdef, typename))
            except Exception as e:
                print(f"  [warn] {typename}: {e}")
        ts.register(add_dict)
        return ts


# =============================================================================
# Style — IEEE-like, LaTeX if available
# =============================================================================
try:
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
        }
    )
    _f, _a = plt.subplots()
    _a.set_title(r"$\alpha$")
    plt.close(_f)
    USE_LATEX = True
except Exception:
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "serif"})
    USE_LATEX = False

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

C = {
    "asv": "#1f77b4",
    "ref": "#d62728",
    "mpc_pred": "#ff7f0e",
    "obs": "#2ca02c",
    "thresh": "#9467bd",
    "surge": "#1f77b4",
    "sway": "#ff7f0e",
    "yaw": "#2ca02c",
    "w_along": "#1f77b4",
    "w_cross": "#d62728",
    "w_head": "#ff7f0e",
    "w_avoid": "#9467bd",
    "solve": "#1f77b4",
}

WEIGHT_LABELS = [
    r"$w_{\mathrm{along}}$" if USE_LATEX else "w_along",
    r"$w_{\mathrm{cross}}$" if USE_LATEX else "w_cross",
    r"$w_{\psi}$" if USE_LATEX else "w_heading",
    r"$w_{\mathrm{input}}$" if USE_LATEX else "w_input",
    r"$w_{u}$" if USE_LATEX else "w_surge",
    r"$w_{v}$" if USE_LATEX else "w_sway",
    r"$w_{r}$" if USE_LATEX else "w_yaw",
    r"$w_{\mathrm{term}}$" if USE_LATEX else "w_terminal",
    r"$w_{\mathrm{avo}}$" if USE_LATEX else "w_avoidance",
]

A_ELL = 95.0
B_ELL = 50.0


# =============================================================================
# Helpers
# =============================================================================
def ns_to_s(ns_array, t0):
    return (np.asarray(ns_array, dtype=np.float64) - t0) / 1e9


def smooth(y, window=31, poly=3):
    y = np.asarray(y)
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def rmse(arr):
    return float(np.sqrt(np.mean(np.asarray(arr, dtype=float) ** 2)))


def _figsize(cols=1, rows=1):
    return (3.39 * cols, 1.8 * rows)


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
        print(f"  Saved {out_dir / name}.{ext}")
    plt.close(fig)


# =============================================================================
# Data extraction
# =============================================================================
def read_bag(bag_path: Path):
    ts = _build_typestore()

    out = {
        "/asv/state/odom": {
            "t": [],
            "x": [],
            "y": [],
            "psi": [],
            "u": [],
            "v": [],
            "r": [],
        },
        "/asv/state/ref": {
            "t": [],
            "x": [],
            "y": [],
            "psi": [],
            "u": [],
            "v": [],
            "r": [],
        },
        "/asv/thrust": {"t": [], "force0": [], "force1": [], "ang0": [], "ang1": []},
        "/asv/state/velocity": {"t": [], "vx": [], "vy": [], "vz": []},
        "/mpc/debug/c_e": {"t": [], "val": []},
        "/mpc/debug/h_e": {"t": [], "val": []},
        "/mpc/debug/a_e": {"t": [], "val": []},
        "/mpc/debug/min_d": {"t": [], "val": []},
        "/mpc/sol_time": {"t": [], "val": []},
        "/mpc/debug/w_log": {"t": [], "rows": []},
        "/mpc/sol_path": {"t": [], "paths": []},
        "/mpc/near_obs": {"t": [], "obs": []},
        "/asv/path_ref": {"t": [], "pts": []},
        "/aitsmc/debug/x": {"t": [], "e": [], "e_i": [], "s": [], "k": [], "u": []},
        "/aitsmc/debug/y": {"t": [], "e": [], "e_i": [], "s": [], "k": [], "u": []},
        "/aitsmc/debug/psi": {"t": [], "e": [], "e_i": [], "s": [], "k": [], "u": []},
    }

    print(f"  Reading bag: {bag_path}")
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            if topic not in out:
                continue
            try:
                msg = ts.deserialize_cdr(rawdata, connection.msgtype)
            except Exception as e:
                print(f"  [warn] deserialize {topic}: {e}")
                continue

            t = timestamp  # nanoseconds (int)

            if topic == "/asv/state/odom":
                q = msg.pose.pose.orientation
                out[topic]["t"].append(t)
                out[topic]["x"].append(msg.pose.pose.position.x)
                out[topic]["y"].append(msg.pose.pose.position.y)
                out[topic]["psi"].append(quat_to_yaw(q.x, q.y, q.z, q.w))
                out[topic]["u"].append(msg.twist.twist.linear.x)
                out[topic]["v"].append(msg.twist.twist.linear.y)
                out[topic]["r"].append(msg.twist.twist.angular.z)

            elif topic == "/asv/state/ref":
                out[topic]["t"].append(t)
                out[topic]["x"].append(msg.x)
                out[topic]["y"].append(msg.y)
                out[topic]["psi"].append(msg.psi)
                out[topic]["u"].append(msg.u)
                out[topic]["v"].append(msg.v)
                out[topic]["r"].append(msg.r)

            elif topic == "/asv/thrust":
                out[topic]["t"].append(t)
                out[topic]["force0"].append(msg.force0)
                out[topic]["force1"].append(msg.force1)
                out[topic]["ang0"].append(msg.ang0)
                out[topic]["ang1"].append(msg.ang1)

            elif topic == "/asv/state/velocity":
                out[topic]["t"].append(t)
                out[topic]["vx"].append(msg.x)
                out[topic]["vy"].append(msg.y)
                out[topic]["vz"].append(msg.z)

            elif topic in (
                "/mpc/debug/c_e",
                "/mpc/debug/h_e",
                "/mpc/debug/a_e",
                "/mpc/debug/min_d",
                "/mpc/sol_time",
            ):
                out[topic]["t"].append(t)
                out[topic]["val"].append(msg.data)

            elif topic == "/mpc/debug/w_log":
                out[topic]["t"].append(t)
                out[topic]["rows"].append(list(msg.data))

            elif topic == "/mpc/sol_path":
                pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
                if pts:
                    out[topic]["t"].append(t)
                    out[topic]["paths"].append(pts)

            elif topic == "/mpc/near_obs":
                obs = [(o.x, o.y, o.v_x, o.v_y) for o in msg.obs_list]
                out[topic]["t"].append(t)
                out[topic]["obs"].append(obs)

            elif topic == "/asv/path_ref":
                pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
                if pts:
                    out[topic]["t"].append(t)
                    out[topic]["pts"].append(pts)

            elif topic in ("/aitsmc/debug/x", "/aitsmc/debug/y", "/aitsmc/debug/psi"):
                out[topic]["t"].append(t)
                out[topic]["e"].append(msg.e)
                out[topic]["e_i"].append(msg.e_i)
                out[topic]["s"].append(msg.s)
                out[topic]["k"].append(msg.k)
                out[topic]["u"].append(msg.u)

    # Convert all timestamp lists to numpy
    for d in out.values():
        if d["t"]:
            d["t"] = np.array(d["t"], dtype=np.float64)
        else:
            d["t"] = np.array([], dtype=np.float64)

    return out


# =============================================================================
# Figure 1 — XY Trajectory
# =============================================================================
def fig_trajectory(data, t0, out_dir, t_avoid=None, t_start=None):
    odom = data["/asv/state/odom"]
    path = data["/asv/path_ref"]
    obs = data["/mpc/near_obs"]
    sol = data["/mpc/sol_path"]
    min_d = data["/mpc/debug/min_d"]

    fig, ax = plt.subplots(figsize=_figsize(2, 2))

    # Reference spline — last published snapshot, drawn as a dotted line
    if path["pts"]:
        sx, sy = zip(*path["pts"][-1])
        ax.plot(
            sx, sy, color=C["ref"], lw=1.2, ls=":", label="Reference path", zorder=2
        )

    ax.plot(
        odom["x"], odom["y"], color=C["asv"], lw=1.2, label="ASV trajectory", zorder=3
    )
    if odom["x"]:
        ax.plot(odom["x"][0], odom["y"][0], "o", color=C["asv"], ms=5, zorder=5)
        ax.plot(odom["x"][-1], odom["y"][-1], "s", color=C["asv"], ms=5, zorder=5)

    # Resolve the snapshot time (ns): user-specified or auto closest-approach
    snap_t_ns = None
    if t_avoid is not None:
        snap_t_ns = t0 + ((t_start or 0.0) + t_avoid) * 1e9
    elif min_d["t"].size:
        snap_t_ns = min_d["t"][int(np.argmin(min_d["val"]))]

    # MPC prediction horizon at the snapshot moment
    if sol["paths"] and snap_t_ns is not None:
        nearest = int(np.argmin(np.abs(sol["t"] - snap_t_ns)))
        px, py = zip(*sol["paths"][nearest])
        snap_t_s = ns_to_s([snap_t_ns], t0)[0]
        ax.plot(
            px,
            py,
            color=C["mpc_pred"],
            lw=1.0,
            ls="-",
            alpha=0.85,
            label=f"MPC horizon (t={snap_t_s:.1f} s)",
            zorder=4,
        )

    # Obstacle positions + velocity arrows at the snapshot moment
    if obs["obs"] and snap_t_ns is not None and obs["t"].size:
        snap_obs_idx = int(np.argmin(np.abs(obs["t"] - snap_t_ns)))
        snap_obs = [o for o in obs["obs"][snap_obs_idx] if abs(o[0]) < 1e5]
        if snap_obs:
            snap_t_s = ns_to_s([snap_t_ns], t0)[0]
            ox = [o[0] for o in snap_obs]
            oy = [o[1] for o in snap_obs]
            ovx = [o[2] for o in snap_obs]
            ovy = [o[3] for o in snap_obs]
            ax.scatter(
                ox,
                oy,
                s=30,
                c=C["obs"],
                alpha=0.85,
                zorder=6,
                label=f"Obstacles at t={snap_t_s:.1f} s",
            )
            ax.quiver(
                ox,
                oy,
                ovx,
                ovy,
                color=C["obs"],
                alpha=0.75,
                scale_units="xy",
                scale=0.1,
                width=0.003,
                zorder=6,
            )

    ax.set_xlabel("$x$ [m]" if USE_LATEX else "x [m]")
    ax.set_ylabel("$y$ [m]" if USE_LATEX else "y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("ASV Trajectory and Reference Path")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_trajectory")


# =============================================================================
# Figure 2 — Tracking errors
# =============================================================================
def fig_tracking_errors(data, t0, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=_figsize(2, 3), sharex=True)

    for ax, topic, ylabel, title, color in [
        (axes[0], "/mpc/debug/c_e", "CTE [m]", "Cross-track Error", C["asv"]),
        (
            axes[1],
            "/mpc/debug/h_e",
            r"$\sin^2(\Delta\psi/2)$" if USE_LATEX else "sin2(dpsi/2)",
            "Heading Error",
            C["ref"],
        ),
        (
            axes[2],
            "/mpc/debug/a_e",
            "Dist to lookahead [m]",
            "Along-track Distance",
            C["obs"],
        ),
    ]:
        d = data[topic]
        if not d["t"].size:
            continue
        t = ns_to_s(d["t"], t0)
        val = np.array(d["val"])
        ax.plot(t, val, color=color, lw=0.7, alpha=0.4)
        ax.plot(
            t,
            smooth(val),
            color=color,
            lw=1.4,
            label=f"RMS={rmse(val):.2f}" if topic != "/mpc/debug/a_e" else None,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if topic != "/mpc/debug/a_e":
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_tracking_errors")


# =============================================================================
# Figure 3 — Body velocities: actual vs MPC reference
# =============================================================================
def fig_velocities(data, t0, out_dir):
    odom = data["/asv/state/odom"]
    ref = data["/asv/state/ref"]
    fig, axes = plt.subplots(3, 1, figsize=_figsize(2, 3), sharex=True)

    # Drop ref samples where u=v=r=0 — these are solver-failure placeholders
    if ref["t"].size:
        valid = ~(
            (np.array(ref["u"]) == 0.0)
            & (np.array(ref["v"]) == 0.0)
            & (np.array(ref["r"]) == 0.0)
        )
        ref_t = ref["t"][valid]
        ref_u = np.array(ref["u"])[valid]
        ref_v = np.array(ref["v"])[valid]
        ref_r = np.array(ref["r"])[valid]
    else:
        ref_t = ref["t"]
        ref_u = ref_v = ref_r = np.array([])

    ref_vals = {"u": ref_u, "v": ref_v, "r": ref_r}

    for i, (key, ylabel, title) in enumerate(
        [
            ("u", "Surge [m/s]", "Surge Velocity"),
            ("v", "Sway [m/s]", "Sway Velocity"),
            ("r", "Yaw rate [rad/s]", "Yaw Rate"),
        ]
    ):
        col = [C["surge"], C["sway"], C["yaw"]][i]
        t_o = ns_to_s(odom["t"], t0)
        t_r = ns_to_s(ref_t, t0)
        axes[i].plot(t_o, odom[key], color=col, lw=0.6, alpha=0.35)
        axes[i].plot(
            t_o, smooth(np.array(odom[key])), color=col, lw=1.4, label="Actual"
        )
        if t_r.size:
            axes[i].plot(
                t_r,
                ref_vals[key],
                color=col,
                lw=1.0,
                ls="--",
                alpha=0.85,
                label="MPC ref",
            )
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_velocities")


# =============================================================================
# Figure 4 — Obstacle distance
# =============================================================================
def fig_obstacle_distance(data, t0, out_dir):
    d = data["/mpc/debug/min_d"]
    if not d["t"].size:
        print("  [skip] No min_d data.")
        return
    fig, ax = plt.subplots(figsize=_figsize(2, 1.6))
    t = ns_to_s(d["t"], t0)
    val = np.array(d["val"])
    ax.fill_between(
        t,
        0,
        B_ELL,
        color=C["thresh"],
        alpha=0.12,
        label=f"Safety bound ({B_ELL:.0f} m)",
    )
    ax.axhline(B_ELL, color=C["thresh"], lw=1.2, ls="--")
    ax.plot(t, val, color=C["asv"], lw=0.7, alpha=0.4)
    ax.plot(
        t,
        smooth(val, 51),
        color=C["asv"],
        lw=1.4,
        label="Min obstacle dist (predicted)",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.set_title("Minimum Predicted Obstacle Distance")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig4_obstacle_distance")


# =============================================================================
# Figure 5 — Adaptive weight scheduling
# =============================================================================
def fig_weight_scheduling(data, t0, out_dir):
    wlog = data["/mpc/debug/w_log"]
    if not wlog["t"].size:
        print("  [skip] No w_log data.")
        return
    rows = np.array(wlog["rows"])
    t_w = ns_to_s(wlog["t"], t0)
    selected = [
        (0, C["w_along"], WEIGHT_LABELS[0]),
        (1, C["w_cross"], WEIGHT_LABELS[1]),
        (2, C["w_head"], WEIGHT_LABELS[2]),
        (8, C["w_avoid"], WEIGHT_LABELS[8]),
    ]
    fig, ax = plt.subplots(figsize=_figsize(2, 1.8))
    for idx, color, label in selected:
        if idx < rows.shape[1]:
            ax.plot(t_w, rows[:, idx], color=color, lw=1.2, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\log_{10}(w)$" if USE_LATEX else "log10(w)")
    ax.set_title("Adaptive Weight Scheduling")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_weight_scheduling")


# =============================================================================
# Figure 6 — MPC solve time
# =============================================================================
def fig_solve_time(data, t0, out_dir):
    d = data["/mpc/sol_time"]
    if not d["t"].size:
        print("  [skip] No sol_time data.")
        return
    t_s = ns_to_s(d["t"], t0)
    vals = np.array(d["val"]) * 1e3  # s → ms
    fig, axes = plt.subplots(1, 2, figsize=_figsize(2, 1.6))
    axes[0].plot(t_s, vals, color=C["solve"], lw=0.6, alpha=0.6)
    axes[0].axhline(
        np.mean(vals), color="k", lw=1.0, ls="--", label=f"Mean {np.mean(vals):.1f} ms"
    )
    axes[0].axhline(
        np.percentile(vals, 95),
        color="r",
        lw=1.0,
        ls=":",
        label=f"95th pct {np.percentile(vals, 95):.1f} ms",
    )
    axes[0].axhline(50.0, color="#d62728", lw=0.8, ls="-.", label="50 ms budget")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Solve time [ms]")
    axes[0].set_title("Solve Time vs Time")
    axes[0].legend(fontsize=7)
    axes[1].hist(vals, bins=60, color=C["solve"], edgecolor="none", alpha=0.8)
    axes[1].axvline(np.mean(vals), color="k", lw=1.2, ls="--")
    axes[1].set_xlabel("Solve time [ms]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Solve Time Distribution")
    fig.tight_layout()
    _save(fig, out_dir, "fig6_solve_time")


# =============================================================================
# Figure 7 — AITSMC: error, sliding surface, adaptive gain (3×3 grid)
# =============================================================================
def fig_aitsmc(data, t0, out_dir):
    channels = [
        ("/aitsmc/debug/x", "Surge ($u$)" if USE_LATEX else "Surge (u)", C["surge"]),
        ("/aitsmc/debug/y", "Sway ($v$)" if USE_LATEX else "Sway (v)", C["sway"]),
        (
            "/aitsmc/debug/psi",
            r"Heading ($\psi$)" if USE_LATEX else "Heading (psi)",
            C["yaw"],
        ),
    ]
    if not any(len(data[k]["t"]) for k, _, _ in channels):
        print("  [skip] No AITSMC debug data.")
        return

    row_keys = ["e", "s", "k"]
    row_ylbls = [
        "$e$" if USE_LATEX else "e",
        "$s$" if USE_LATEX else "s",
        "$k$" if USE_LATEX else "k",
    ]
    row_titles = [
        "Tracking error",
        "Sliding surface",
        "Adaptive gain $k$" if USE_LATEX else "Adaptive gain k",
    ]

    fig, axes = plt.subplots(3, 3, figsize=_figsize(2, 3), sharex=True)

    for col, (topic, col_title, color) in enumerate(channels):
        d = data[topic]
        if not len(d["t"]):
            continue
        t = ns_to_s(d["t"], t0)
        for row, key in enumerate(row_keys):
            arr = np.array(d[key])
            axes[row, col].plot(t, arr, color=color, lw=0.6, alpha=0.35)
            axes[row, col].plot(t, smooth(arr), color=color, lw=1.3)
            if key in ("e", "s"):
                axes[row, col].axhline(0, color="k", lw=0.7, ls="--")
            if row == 0:
                axes[row, col].set_title(col_title)
            if col == 0:
                axes[row, col].set_ylabel(row_ylbls[row])
        axes[-1, col].set_xlabel("Time [s]")

    fig.tight_layout()
    _save(fig, out_dir, "fig7_aitsmc")


# =============================================================================
# Figure 8 — Azimuth thruster commands
# =============================================================================
def fig_thrust(data, t0, out_dir):
    thr = data["/asv/thrust"]
    if not len(thr["t"]):
        print("  [skip] No /asv/thrust data.")
        return
    fig, axes = plt.subplots(2, 2, figsize=_figsize(2, 2.6), sharex=True)
    t_thr = ns_to_s(thr["t"], t0)

    for ax, key, label, color in [
        (axes[0, 0], "force0", "Thruster 0", C["surge"]),
        (axes[0, 1], "force1", "Thruster 1", C["sway"]),
    ]:
        raw = np.array(thr[key]) / 1e3
        ax.plot(t_thr, raw, color=color, lw=0.6, alpha=0.35)
        ax.plot(t_thr, smooth(raw), color=color, lw=1.4)
        ax.set_ylabel("Force [kN]")
        ax.set_title(label)

    for ax, key, label, color in [
        (axes[1, 0], "ang0", "Thruster 0", C["surge"]),
        (axes[1, 1], "ang1", "Thruster 1", C["sway"]),
    ]:
        raw = np.rad2deg(np.array(thr[key]))
        ax.plot(t_thr, raw, color=color, lw=0.6, alpha=0.35)
        ax.plot(t_thr, smooth(raw), color=color, lw=1.4)
        ax.set_ylabel("Angle [deg]")
        ax.set_xlabel("Time [s]")

    fig.suptitle("Azimuth Thruster Commands", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_thrust")


# =============================================================================
# Stats summary
# =============================================================================
def print_stats(data):
    print("\n" + "=" * 52)
    print("  PERFORMANCE SUMMARY")
    print("=" * 52)
    for label, topic in [
        ("CTE [m]", "/mpc/debug/c_e"),
        ("Heading err", "/mpc/debug/h_e"),
        ("Min obs dist [m]", "/mpc/debug/min_d"),
    ]:
        v = np.array(data[topic]["val"])
        if v.size:
            print(
                f"  {label:<18}  RMS={rmse(v):.3f}  max={v.max():.3f}  min={v.min():.3f}"
            )
    st = np.array(data["/mpc/sol_time"]["val"]) * 1e3
    if st.size:
        print(
            f"  Solve time [ms]    mean={st.mean():.2f}  "
            f"95pct={np.percentile(st, 95):.2f}  max={st.max():.2f}"
        )
    odom = data["/asv/state/odom"]
    if odom["t"].size > 1:
        dur = (odom["t"][-1] - odom["t"][0]) / 1e9
        print(f"  Bag duration:      {dur:.1f} s")
    print("=" * 52 + "\n")


# =============================================================================
# Time-window filtering
# =============================================================================

# Inclusive time window in seconds relative to the bag start (t0).
# Set to None to disable clipping on that end.
T_START: float | None = None  # e.g. 10.0  → discard the first 10 s
T_END: float | None = None  # e.g. 120.0 → discard everything after 120 s

# Time (seconds relative to t0) for the obstacle/MPC-horizon snapshot in
# fig_trajectory.  None → auto-select the closest-approach moment.
T_AVOID: float | None = None


def _clip_topic(d: dict, t_lo: float, t_hi: float) -> dict:
    """Return a copy of topic dict *d* keeping only samples in [t_lo, t_hi] ns."""
    t = d["t"]
    if t.size == 0:
        return d
    mask = np.ones(t.size, dtype=bool)
    if t_lo is not None:
        mask &= t >= t_lo
    if t_hi is not None:
        mask &= t <= t_hi
    if mask.all():
        return d
    clipped = {"t": t[mask]}
    for key, val in d.items():
        if key == "t":
            continue
        if isinstance(val, np.ndarray) and val.shape and val.shape[0] == t.size:
            clipped[key] = val[mask]
        elif isinstance(val, list) and len(val) == t.size:
            clipped[key] = [val[i] for i in range(len(val)) if mask[i]]
        else:
            clipped[key] = val
    return clipped


def clip_data(
    data: dict, t0: float, t_start: float | None, t_end: float | None
) -> dict:
    """Filter all topics to [t_start, t_end] (seconds relative to t0)."""
    if t_start is None and t_end is None:
        return data
    t_lo = (t0 + t_start * 1e9) if t_start is not None else None
    t_hi = (t0 + t_end * 1e9) if t_end is not None else None
    return {topic: _clip_topic(d, t_lo, t_hi) for topic, d in data.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot ASV MPC rosbag results")
    parser.add_argument(
        "bag_dir",
        nargs="?",
        default=".",
        help="Path to rosbag2 directory (contains .db3)",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        metavar="S",
        help="Start of plot window in seconds (relative to bag start)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        metavar="S",
        help="End of plot window in seconds (relative to bag start)",
    )
    parser.add_argument(
        "--t-avoid",
        type=float,
        default=None,
        metavar="S",
        help="Time (s, relative to bag start) for obstacle/MPC snapshot in XY plot "
        "(default: auto = closest-approach moment)",
    )
    args = parser.parse_args()

    # Module-level defaults are overridden by CLI when provided
    t_start = args.t_start if args.t_start is not None else T_START
    t_end = args.t_end if args.t_end is not None else T_END
    t_avoid = args.t_avoid if args.t_avoid is not None else T_AVOID

    bag_dir = Path(args.bag_dir).resolve()
    if not bag_dir.exists():
        sys.exit(f"Bag directory not found: {bag_dir}")

    lo_str = f"{t_start:.0f}s" if t_start is not None else "start"
    hi_str = f"{t_end:.0f}s" if t_end is not None else "end"
    out_dir = bag_dir / f"figures_{lo_str}_to_{hi_str}"
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}")

    print("\nReading bag data...")
    data = read_bag(bag_dir)

    # Global t0: earliest non-empty timestamp
    t0 = min(
        d["t"][0]
        for d in data.values()
        if isinstance(d["t"], np.ndarray) and d["t"].size > 0
    )
    print(f"  t0 = {t0:.0f} ns")

    if t_start is not None or t_end is not None:
        lo_disp = f"{t_start:.1f} s" if t_start is not None else "start"
        hi_disp = f"{t_end:.1f} s" if t_end is not None else "end"
        print(f"  Applying time window: [{lo_disp}, {hi_disp}]")
        data = clip_data(data, t0, t_start, t_end)

    print_stats(data)

    print("Generating figures...")
    fig_trajectory(data, t0, out_dir, t_avoid=t_avoid, t_start=t_start)
    fig_tracking_errors(data, t0, out_dir)
    fig_velocities(data, t0, out_dir)
    fig_obstacle_distance(data, t0, out_dir)
    fig_weight_scheduling(data, t0, out_dir)
    fig_solve_time(data, t0, out_dir)
    fig_aitsmc(data, t0, out_dir)
    fig_thrust(data, t0, out_dir)

    print(f"\nAll figures written to: {out_dir}")


if __name__ == "__main__":
    main()
