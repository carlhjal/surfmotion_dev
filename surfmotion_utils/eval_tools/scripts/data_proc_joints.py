import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from scipy.signal import savgol_filter
import seaborn as sns  # Optional, but gives nice overlays and color
from scipy.spatial.transform import Rotation as R
import math
from tabulate import tabulate
# --- Config ---
subdirs = {
    "ur20": {"cart": "cart_ur20", "pilz": "pilz_ur20", "servo": "servo_ur20"},
    "fanuc": {"cart": "cart_fanuc", "pilz": "pilz_fanuc", "servo": "servo_fanuc"},
    "ur5": {"cart": "cart_ur5", "pilz": "pilz_ur5", "servo": "servo_ur5"},
    "kuka": {"cart": "cart_kuka", "pilz": "pilz_kuka", "servo": "servo_kuka"}
}

robots = ["ur20", "fanuc", "ur5", "kuka"]
planners = ["cart", "pilz", "servo"]
planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
window_length = 25   # Must be odd!
polyorder = 3

# plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})

plt.rcParams.update({
    "font.size": 12,          # Controls default text size
    "axes.titlesize": 16,     # Axes title
    "axes.labelsize": 16,     # Axes labels
    "xtick.labelsize": 16,    # Tick labels
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "figure.titlesize": 13
})

def quat_to_angular_vel(q, dt):
    """
    q: (N, 4) array of quaternions (x, y, z, w)
    dt: scalar (assumed uniform spacing)
    Returns: (N-1, 3) angular velocity in rad/s
    """
    # Use scipy Rotation for batch difference
    r1 = R.from_quat(q[:-1])
    r2 = R.from_quat(q[1:])
    rel_rot = r2 * r1.inv()
    angvec = rel_rot.as_rotvec()  # axis-angle (radians)
    # Divide by timestep to get angular velocity
    ang_vel = angvec / dt
    return ang_vel

def load_joint_csv(path):
    df = pd.read_csv(path)
    # Defensive: in case columns are not sorted
    df = df.sort_values(['joint', 'time'])
    return df

def compute_derivatives(t, pos, window_length, polyorder):
    """Returns (vel, acc, jerk) using Savitzky-Golay, with edge handling"""
    # Uniformly spaced t? If not, will be approximate.
    dt = np.mean(np.diff(t))
    if len(pos) < window_length:
        return None, None, None  # Not enough data
    vel = savgol_filter(pos, window_length, polyorder, deriv=1, delta=dt, mode="interp")
    acc = savgol_filter(pos, window_length, polyorder, deriv=2, delta=dt, mode="interp")
    jerk = savgol_filter(pos, window_length, polyorder, deriv=3, delta=dt, mode="interp")
    return vel, acc, jerk

# Collect all data
all_rms_stats = {robot: {planner: {} for planner in planners} for robot in robots}

for robot in robots:
    for planner in planners:
        data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
        joint_files = sorted(glob.glob(os.path.join(data_dir, "joint*")))
        if not joint_files:
            print(f"WARNING: No joint files found for {robot}-{planner}")
            continue

        # Discover all unique joint names for this planner (across all runs)
        joints = set()
        for f in joint_files:
            df = load_joint_csv(f)
            joints.update(df["joint"].unique())
        joints = sorted(list(joints))

        for joint in joints:
            all_rms_stats[robot][planner][joint] = []

        # Loop over runs (joint files)
        for jf in joint_files:
            df = load_joint_csv(jf)
            for joint in joints:
                jdata = df[df["joint"] == joint]
                if len(jdata) < window_length:
                    continue
                t = jdata["time"].values
                pos = jdata["position"].values
                # Skip if weirdly nonmonotonic
                if np.any(np.diff(t) <= 0):
                    continue
                vel, acc, jerk = compute_derivatives(t, pos, window_length, polyorder)
                # Compute RMS for this run/joint
                if vel is None: continue
                rms_vel = np.sqrt(np.mean(vel ** 2))
                rms_acc = np.sqrt(np.mean(acc ** 2))
                rms_jerk = np.sqrt(np.mean(jerk ** 2))
                all_rms_stats[robot][planner][joint].append((rms_vel, rms_acc, rms_jerk))

for robot in robots:
    for planner in planners:
        data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
        eef_files = sorted(glob.glob(os.path.join(data_dir, "eef*")))
        eef_rms = []
        for ef in eef_files:
            df = pd.read_csv(ef)
            t = df["time"].values
            # Assume columns are ['x', 'y', 'z']
            pos = df[["x", "y", "z"]].values
            if len(pos) < window_length: continue
            dt = np.mean(np.diff(t))
            vel = savgol_filter(pos, window_length, polyorder, deriv=1, delta=dt, mode="interp", axis=0)
            acc = savgol_filter(pos, window_length, polyorder, deriv=2, delta=dt, mode="interp", axis=0)
            jerk = savgol_filter(pos, window_length, polyorder, deriv=3, delta=dt, mode="interp", axis=0)
            vel_mag = np.linalg.norm(vel, axis=1)
            acc_mag = np.linalg.norm(acc, axis=1)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            rms_vel = np.sqrt(np.mean(vel_mag ** 2))
            rms_acc = np.sqrt(np.mean(acc_mag ** 2))
            rms_jerk = np.sqrt(np.mean(jerk_mag ** 2))
            eef_rms.append((rms_vel, rms_acc, rms_jerk))
        # Add to all_rms_stats with a special "joint" name
        if "EEF_Cartesian" not in all_rms_stats[robot][planner]:
            all_rms_stats[robot][planner]["EEF_Cartesian"] = []
        all_rms_stats[robot][planner]["EEF_Cartesian"].extend(eef_rms)

        window_length = 25
        eef_ang_rms = []
        for ef in eef_files:
            df = pd.read_csv(ef)
            t = df["time"].values
            quat = df[["qx", "qy", "qz", "qw"]].values
            if len(quat) < window_length: continue
            dt = np.mean(np.diff(t))
            # Compute angular velocity using quaternion finite difference
            ang_vel = quat_to_angular_vel(quat, dt)  # shape (N-1, 3)
            # Savitzky-Golay filter for derivatives
            if len(ang_vel) < window_length: continue
            ang_vel_filt = savgol_filter(ang_vel, window_length, polyorder, deriv=0, axis=0)
            ang_acc = savgol_filter(ang_vel, window_length, polyorder, deriv=1, delta=dt, axis=0)
            ang_jerk = savgol_filter(ang_vel, window_length, polyorder, deriv=2, delta=dt, axis=0)
            ang_vel_mag = np.linalg.norm(ang_vel_filt, axis=1)
            ang_acc_mag = np.linalg.norm(ang_acc, axis=1)
            ang_jerk_mag = np.linalg.norm(ang_jerk, axis=1)
            # Convert from rad/s to deg/s for reporting
            rms_ang_vel = np.sqrt(np.mean(np.square(ang_vel_mag))) * (180.0 / np.pi)
            rms_ang_acc = np.sqrt(np.mean(np.square(ang_acc_mag))) * (180.0 / np.pi)
            rms_ang_jerk = np.sqrt(np.mean(np.square(ang_jerk_mag))) * (180.0 / np.pi)
            eef_ang_rms.append((rms_ang_vel, rms_ang_acc, rms_ang_jerk))
        # Store in joint dict as "EEF_Angular"
        if "EEF_Angular" not in all_rms_stats[robot][planner]:
            all_rms_stats[robot][planner]["EEF_Angular"] = []
        all_rms_stats[robot][planner]["EEF_Angular"].extend(eef_ang_rms)

# --- Plotting ---


# metric_names = ["Velocity (RMS)", "Acceleration (RMS)", "Jerk (RMS)"]
# metric_units = ["[m/s]", "[m/s²]", "[m/s³]"]

# robots = ["ur20", "fanuc", "ur5", "kuka"]
# planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
# metrics = ["Velocity (RMS)", "Acceleration (RMS)", "Jerk (RMS)"]
# metric_units_cart = ["[m/s]", "[m/s²]", "[m/s³]"]
# metric_units_ang = ["[deg/s]", "[deg/s²]", "[deg/s³]"]  # Or [rad/s] if that's your units

# robot_labels = ["UR20", "FANUC", "UR5", "KUKA"]
# colors = ["C0", "C1", "C2", "C3", "gray"]
# metric_idx = 2  # 0=velocity, 1=acceleration, 2=jerk

# planner_colors = {
#     "cart": "#8D5A99",   # Slate blue
#     "pilz": "#e76f51",   # Burnt orange/burgundy
#     "servo": "#2186a5"   # Deep purple
# }
# robots = ["ur20", "fanuc", "ur5", "kuka"]
# planners = ["cart", "pilz", "servo"]
# planner_labels = ["Cartesian", "Pilz", "Servo"]
# color_list = [planner_colors[p] for p in planners]

# bar_width = 0.7
# fontsize_title = 16
# fontsize_label = 14
# fontsize_tick = 14
# fontsize_val = 10

# def get_avg_scores(category):
#     avg_plain, avg_adj = [], []
#     for planner in planners:
#         plain_vals = []
#         adj_vals = []
#         for robot in robots:
#             if category == "cart":
#                 vals = [v for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#             elif category == "ang":
#                 vals = [v for v in all_rms_stats[robot][planner].get("EEF_Angular", []) if v is not None]
#             elif category == "joint":
#                 joint_names = [jn for jn in all_rms_stats[robot][planner] if not jn.startswith("EEF")]
#                 vals = []
#                 for jn in joint_names:
#                     vals += [v for v in all_rms_stats[robot][planner][jn] if v is not None]
#             else:
#                 vals = []

#             if vals:
#                 plain = np.mean([v[2] for v in vals])
#                 plain_vals.append(plain)
#                 mean_vel = np.mean([v[0] for v in vals])
#                 adj = plain / mean_vel if mean_vel > 1e-8 else np.nan
#                 adj_vals.append(adj)
#         avg_plain.append(np.nanmean(plain_vals) if plain_vals else np.nan)
#         avg_adj.append(np.nanmean(adj_vals) if adj_vals else np.nan)
#     return avg_plain, avg_adj

# plain_cart, adj_cart = get_avg_scores("cart")
# plain_ang, adj_ang = get_avg_scores("ang")
# plain_joint, adj_joint = get_avg_scores("joint")

# x = np.arange(len(planner_labels))

# fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex='col')

# metric_titles = [
#     "Mean Translational Smoothness",
#     "Mean Angular Smoothness",
#     "Mean Joint-space Smoothness"
# ]
# ylabels = [
#     "Mean EEF Translational\nSmoothness [m/s³]",
#     "Mean EEF Angular\nSmoothness [deg/s³]",
#     "Mean Joint-space\nSmoothness [rad/s³]"
# ]
# adj_ylabels = [
#     "Mean Velocity-Adjusted\nTranslational smoothness [1/s²]",
#     "Mean Velocity-Adjusted\nAngular smoothness [1/s²]",
#     "Mean Velocity-Adjusted\nJoint-space smoothness [1/s²]"
# ]

# data_sets = [(plain_cart, adj_cart), (plain_ang, adj_ang), (plain_joint, adj_joint)]

# for row in range(3):
#     for col in range(2):
#         vals = data_sets[row][col]
#         ax = axes[row, col]
#         ax.set_xticks(x)
#         ax.set_xticklabels(planner_labels, fontsize=fontsize_tick)
#         ax.tick_params(axis='y', labelsize=fontsize_tick)
#         ax.set_axisbelow(True)  # This puts grid BELOW everything else
#         # Draw grid (solid, and at lower zorder)
#         ax.grid(True, which="both", axis="y", linestyle='-', alpha=0.7, zorder=0)
#         # Bars at higher zorder
#         bars = ax.bar(x, vals, color=color_list, width=bar_width, edgecolor="black", zorder=3)
#         if col == 0:
#             ax.set_ylabel(ylabels[row], fontsize=fontsize_label)
#             ax.set_title(metric_titles[row], fontsize=fontsize_title, pad=8)
#         else:
#             ax.set_ylabel(adj_ylabels[row], fontsize=fontsize_label)
#             ax.set_title("Velocity-adjusted", fontsize=fontsize_title, pad=8)


# fig.suptitle("Average Smoothness Scores per Planner", fontsize=20)
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.show()


# robots = ["ur20", "fanuc", "ur5", "kuka"]
# planners = ["cart", "pilz", "servo"]
# planner_labels = ["Cartesian", "Pilz", "Servo"]
# colors = ["C0", "C1", "C2"]

# # --- Helper to get scores ---
# def get_avg_scores(category):
#     avg_plain, avg_adj = [], []
#     for planner in planners:
#         plain_vals = []
#         adj_vals = []
#         for robot in robots:
#             if category == "cart":
#                 vals = [v for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#             elif category == "ang":
#                 vals = [v for v in all_rms_stats[robot][planner].get("EEF_Angular", []) if v is not None]
#             elif category == "joint":
#                 joint_names = [jn for jn in all_rms_stats[robot][planner] if not jn.startswith("EEF")]
#                 vals = []
#                 for jn in joint_names:
#                     vals += [v for v in all_rms_stats[robot][planner][jn] if v is not None]
#             else:
#                 vals = []

#             if vals:
#                 # Plain RMS jerk
#                 plain = np.mean([v[2] for v in vals])
#                 plain_vals.append(plain)
#                 # Velocity-adjusted: RMS jerk / RMS velocity
#                 mean_vel = np.mean([v[0] for v in vals])
#                 if mean_vel > 1e-8:
#                     adj = plain / mean_vel
#                     adj_vals.append(adj)
#                 else:
#                     adj_vals.append(np.nan)
#         avg_plain.append(np.nanmean(plain_vals) if plain_vals else np.nan)
#         avg_adj.append(np.nanmean(adj_vals) if adj_vals else np.nan)
#     return avg_plain, avg_adj

# # --- Data for each metric ---
# plain_cart, adj_cart = get_avg_scores("cart")
# plain_ang, adj_ang = get_avg_scores("ang")
# plain_joint, adj_joint = get_avg_scores("joint")

# bar_width = 0.8  # Wider bars for clarity
# x = np.arange(len(planner_labels))

# fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex='col')

# fontsize_title = 15
# fontsize_label = 12
# fontsize_tick = 12
# fontsize_val = 12

# # -- Titles for columns --
# axes[0,0].set_title("RMS Jerk", fontsize=fontsize_title+1)
# axes[0,1].set_title("Velocity-adjusted\n(RMS jerk / RMS vel)", fontsize=fontsize_title+1)

# # --- Translational (Top row) ---
# axes[0,0].bar(x, plain_cart, color=colors)
# axes[0,0].set_ylabel("Translational\n[m/s³]", fontsize=fontsize_label)
# axes[0,0].set_xticks(x)
# axes[0,0].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# axes[0,1].bar(x, adj_cart, color=colors)
# axes[0,1].set_xticks(x)
# axes[0,1].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# # --- Angular (Middle row) ---
# axes[1,0].bar(x, plain_ang, color=colors)
# axes[1,0].set_ylabel("Angular\n[deg/s³]", fontsize=fontsize_label)
# axes[1,0].set_xticks(x)
# axes[1,0].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# axes[1,1].bar(x, adj_ang, color=colors)
# axes[1,1].set_xticks(x)
# axes[1,1].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# # --- Joint-space (Bottom row) ---
# axes[2,0].bar(x, plain_joint, color=colors)
# axes[2,0].set_ylabel("Joint-space\n[rad/s³]", fontsize=fontsize_label)
# axes[2,0].set_xticks(x)
# axes[2,0].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# axes[2,1].bar(x, adj_joint, color=colors)
# axes[2,1].set_xticks(x)
# axes[2,1].set_xticklabels(planner_labels, fontsize=fontsize_tick)

# for axrow in axes:
#     for ax in axrow:
#         ax.tick_params(axis='y', labelsize=fontsize_tick)

# fig.suptitle("Average Smoothness Scores per Planner\n(Left: RMS jerk, Right: Velocity-adjusted)", fontsize=18)
# plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# plt.show()




# # Prepare headers
# headers = [
#     "Metric",
#     "Cartesian", "Pilz", "Servo",
#     "Cartesian (adj.)", "Pilz (adj.)", "Servo (adj.)"
# ]

# # Format data
# table = []

# # Translational
# row = [
#     "Translational [m/s$^3$]",
#     *(f"{v:.3f}" for v in plain_cart),
#     *(f"{v:.2f}" for v in adj_cart),
# ]
# table.append(row)

# # Angular
# row = [
#     "Angular [deg/s$^3$]",
#     *(f"{v:.2f}" for v in plain_ang),
#     *(f"{v:.2f}" for v in adj_ang),
# ]
# table.append(row)

# # Joint-space
# row = [
#     "Joint-space [rad/s$^3$]",
#     *(f"{v:.3f}" for v in plain_joint),
#     *(f"{v:.2f}" for v in adj_joint),
# ]
# table.append(row)

# # Print LaTeX table
# latex = tabulate(table, headers=headers, tablefmt="latex")
# print(latex)

# headers = [
#     "Metric",
#     "Cartesian [m/s$^3$]", "Pilz [m/s$^3$]", "Servo [m/s$^3$]",
#     "Cartesian (adj.) [1/s$^2$]", "Pilz (adj.) [1/s$^2$]", "Servo (adj.) [1/s$^2$]"
# ]
# table = [
#     ["Translational RMS jerk", f"{plain_cart[0]:.3f}", f"{plain_cart[1]:.3f}", f"{plain_cart[2]:.3f}",
#      f"{adj_cart[0]:.2f}", f"{adj_cart[1]:.2f}", f"{adj_cart[2]:.2f}"],
#     ["Angular RMS jerk", f"{plain_ang[0]:.2f}", f"{plain_ang[1]:.2f}", f"{plain_ang[2]:.2f}",
#      f"{adj_ang[0]:.2f}", f"{adj_ang[1]:.2f}", f"{adj_ang[2]:.2f}"],
#     ["Joint-space RMS jerk", f"{plain_joint[0]:.3f}", f"{plain_joint[1]:.3f}", f"{plain_joint[2]:.3f}",
#      f"{adj_joint[0]:.2f}", f"{adj_joint[1]:.2f}", f"{adj_joint[2]:.2f}"]
# ]

# from tabulate import tabulate
# print(tabulate(table, headers=headers, tablefmt="latex"))


# planners = ["cart", "pilz", "servo"]
# planner_labels = ["Cartesian", "Pilz", "Servo"]

# # Get average across all robots for each planner

# robots = ["ur20", "fanuc", "ur5", "kuka"]
# planners = ["cart", "pilz", "servo"]
# planner_labels = ["Cartesian", "Pilz", "Servo"]
# colors = ["C0", "C1", "C2"]

# robots = ["ur20", "fanuc", "ur5", "kuka"]
# planners = ["cart", "pilz", "servo"]
# planner_labels = ["Cartesian", "Pilz", "Servo"]
# colors = ["C0", "C1", "C2"]

# # Translational
# avg_cart = []
# for planner in planners:
#     vals = []
#     for robot in robots:
#         jerk_vals = [v[2] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#         if jerk_vals:
#             vals.append(np.mean(jerk_vals))
#     avg_cart.append(np.nanmean(vals) if vals else np.nan)
# # Angular
# avg_ang = []
# for planner in planners:
#     vals = []
#     for robot in robots:
#         jerk_vals = [v[2] for v in all_rms_stats[robot][planner].get("EEF_Angular", []) if v is not None]
#         if jerk_vals:
#             vals.append(np.mean(jerk_vals))
#     avg_ang.append(np.nanmean(vals) if vals else np.nan)
# # Joint-space
# avg_joint = []
# for planner in planners:
#     vals = []
#     for robot in robots:
#         joint_names = [jn for jn in all_rms_stats[robot][planner] if not jn.startswith("EEF")]
#         jerks = []
#         for jn in joint_names:
#             jerks += [v[2] for v in all_rms_stats[robot][planner][jn] if v is not None]
#         if jerks:
#             vals.append(np.mean(jerks))
#     avg_joint.append(np.nanmean(vals) if vals else np.nan)

# # --- Multi-panel (vertical) plot ---
# fig, axes = plt.subplots(3, 1, figsize=(7, 11))  # A4 portrait: 210mm x 297mm ~ 8.3 x 11.7 inches

# fontsize_title = 16
# fontsize_label = 13
# fontsize_tick = 12
# fontsize_val = 12

# # Translational
# axes[0].bar(planner_labels, avg_cart, color=colors)
# axes[0].set_ylabel("Avg EEF Translational\nSmoothness [m/s³]", fontsize=fontsize_label)
# axes[0].set_title("Translational RMS Jerk", fontsize=fontsize_title)
# axes[0].tick_params(axis='x', labelsize=fontsize_tick)
# axes[0].tick_params(axis='y', labelsize=fontsize_tick)
# for i, v in enumerate(avg_cart):
#     if not np.isnan(v):
#         axes[0].text(i, v + 0.001*max(avg_cart), f"{v:.3f}", ha='center', va='bottom', fontsize=fontsize_val)

# # Angular
# axes[1].bar(planner_labels, avg_ang, color=colors)
# axes[1].set_ylabel("Avg EEF Angular\nSmoothness [deg/s³]", fontsize=fontsize_label)
# axes[1].set_title("Angular RMS Jerk", fontsize=fontsize_title)
# axes[1].tick_params(axis='x', labelsize=fontsize_tick)
# axes[1].tick_params(axis='y', labelsize=fontsize_tick)
# for i, v in enumerate(avg_ang):
#     if not np.isnan(v):
#         axes[1].text(i, v + 0.001*max(avg_ang), f"{v:.3f}", ha='center', va='bottom', fontsize=fontsize_val)

# # Joint-space
# axes[2].bar(planner_labels, avg_joint, color=colors)
# axes[2].set_ylabel("Avg Joint-space\nSmoothness [rad/s³]", fontsize=fontsize_label)
# axes[2].set_title("Joint-space Mean RMS Jerk", fontsize=fontsize_title)
# axes[2].tick_params(axis='x', labelsize=fontsize_tick)
# axes[2].tick_params(axis='y', labelsize=fontsize_tick)
# for i, v in enumerate(avg_joint):
#     if not np.isnan(v):
#         axes[2].text(i, v + 0.001*max(avg_joint), f"{v:.3f}", ha='center', va='bottom', fontsize=fontsize_val)

# fig.suptitle("Average Smoothness Scores per Planner", fontsize=18)
# plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# # For thesis-quality: save as high-res PDF or PNG
# plt.savefig("smoothness_scores_a4.pdf", bbox_inches="tight")
# plt.show()

































# for planner in planners:
#     # Collect mean RMS jerk for each robot
#     jerk_vals = []
#     for robot in robots:
#         vals = [v[2] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#         jerk_vals.append(np.mean(vals) if vals else np.nan)
#     avg_jerk = np.nanmean(jerk_vals)
#     bar_labels = robot_labels + ["Average"]
#     bar_values = jerk_vals + [avg_jerk]

#     # Plot
#     plt.figure(figsize=(8, 6))
#     plt.bar(bar_labels, bar_values, color=["C0", "C1", "C2", "C3", "gray"])
#     plt.ylabel("Smoothness (RMS jerk) [m/s³]")
#     plt.title(f"EEF RMS Jerk (Composite Smoothness)\n{planner_labels[planner]} Planner")
#     for i, v in enumerate(bar_values):
#         if not np.isnan(v):
#             plt.text(i, v + 0.001 * max(bar_values), f"{v:.3f}", ha='center', va='bottom', fontsize=12)
#     plt.tight_layout()
#     plt.show()
    
# for planner in planners:
#     val_scores = []
#     for robot in robots:
#         jerk_list = [v[2] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#         vel_list = [v[0] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#         if jerk_list and vel_list and np.mean(vel_list) > 1e-8:
#             score = np.mean(jerk_list) / np.mean(vel_list)
#         else:
#             score = np.nan
#         val_scores.append(score)
#     avg_score = np.nanmean(val_scores)
#     bar_labels = robot_labels + ["Average"]
#     bar_values = val_scores + [avg_score]

#     plt.figure(figsize=(8, 6))
#     plt.bar(bar_labels, bar_values, color=["C0", "C1", "C2", "C3", "gray"])
#     plt.ylabel("Velocity-adjusted Smoothness (RMS jerk / RMS vel) [1/s²]")
#     plt.title(f"EEF Velocity-adjusted Composite Smoothness\n{planner_labels[planner]} Planner")
#     for i, v in enumerate(bar_values):
#         if not np.isnan(v):
#             plt.text(i, v + 0.001 * max(bar_values), f"{v:.3f}", ha='center', va='bottom', fontsize=12)
#     plt.tight_layout()
#     plt.show()
# for planner in planners:
#     mean_joint_jerks = []
#     for robot in robots:
#         # Get all joints except special EEF keys
#         joint_names = [jn for jn in all_rms_stats[robot][planner] if not jn.startswith("EEF")]
#         # Collect all jerks
#         jerks = []
#         for jn in joint_names:
#             jerks += [v[2] for v in all_rms_stats[robot][planner][jn] if v is not None]
#         mean_joint_jerks.append(np.mean(jerks) if jerks else np.nan)
#     avg_joint_jerk = np.nanmean(mean_joint_jerks)
#     bar_labels = robot_labels + ["Average"]
#     bar_values = mean_joint_jerks + [avg_joint_jerk]

#     plt.figure(figsize=(8, 6))
#     plt.bar(bar_labels, bar_values, color=["C0", "C1", "C2", "C3", "gray"])
#     plt.ylabel("Joint-space Smoothness (Mean RMS jerk) [rad/s³]")
#     plt.title(f"Joint-space Mean RMS Jerk\n{planner_labels[planner]} Planner")
#     for i, v in enumerate(bar_values):
#         if not np.isnan(v):
#             plt.text(i, v + 0.001 * max(bar_values), f"{v:.3f}", ha='center', va='bottom', fontsize=12)
#     plt.tight_layout()
#     plt.show()

# planner = "cart"
# robots = ["ur20", "fanuc", "ur5", "kuka"]
# robot_labels = [r.upper() for r in robots]
# metrics = ["Velocity", "Acceleration", "Jerk"]
# metric_units_cart = ["[m/s]", "[m/s²]", "[m/s³]"]
# metric_units_ang = ["[deg/s]", "[deg/s²]", "[deg/s³]"]


# for planner in planners:
#     fig, axes = plt.subplots(3, 2, figsize=(12, 12))
#     for i, metric in enumerate(metrics):
#         # Translational
#         means = []
#         for robot in robots:
#             vals = [v[i] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#             means.append(np.mean(vals) if vals else np.nan)
#         axes[i, 0].bar(robot_labels, means, capsize=8, edgecolor="black", linewidth=0.7)
#         axes[i, 0].set_ylabel(f"{metric} {metric_units_cart[i]}")
#         axes[i, 0].set_title(f"Translational {metric}")

#         # Angular
#         means = []
#         for robot in robots:
#             vals = [v[i] for v in all_rms_stats[robot][planner].get("EEF_Angular", []) if v is not None]
#             means.append(np.mean(vals) if vals else np.nan)
#         axes[i, 1].bar(robot_labels, means, capsize=8, edgecolor="black", linewidth=0.7)
#         axes[i, 1].set_ylabel(f"{metric} {metric_units_ang[i]}")
#         axes[i, 1].set_title(f"Angular {metric}")

#     # Set main title and tidy up
#     fig.suptitle(f"Mean EEF RMS Translational and Angular Metrics ({planner_labels[planner]})", fontsize=22)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

    
# metrics = ["Velocity", "Acceleration", "Jerk"]
# metric_units_cart = ["[m/s]", "[m/s²]", "[m/s³]"]
# metric_units_ang = ["[deg/s]", "[deg/s²]", "[deg/s³]"]

# fig, axes = plt.subplots(3, 2, figsize=(12, 12))
# planner = "cart"
# for planner in planners:
#     for i, metric in enumerate(metrics):
#         # Translational
#         means = []
#         stds = []
#         for robot in robots:
#             vals = [v[i] for v in all_rms_stats[robot][planner].get("EEF_Cartesian", []) if v is not None]
#             means.append(np.mean(vals) if vals else np.nan)
#             # stds.append(np.std(vals) if vals else 0.0)
#         axes[i, 0].bar(robot_labels, means, capsize=8)
#         axes[i, 0].set_ylabel(f"{metric} {metric_units_cart[i]}")
#         axes[i, 0].set_title(f"Translational {metric}")

#         # Angular
#         means = []
#         stds = []
#         for robot in robots:
#             vals = [v[i] for v in all_rms_stats[robot][planner].get("EEF_Angular", []) if v is not None]
#             means.append(np.mean(vals) if vals else np.nan)
#             # stds.append(np.std(vals) if vals else 0.0)
#         axes[i, 1].bar(robot_labels, means, capsize=8)
#         axes[i, 1].set_ylabel(f"{metric} {metric_units_ang[i]}")
#         axes[i, 1].set_title(f"Angular {metric}")

#     plt.suptitle(f"EEF RMS Translational and Angular Metrics (MoveIt Cartesian Planner)", fontsize=18)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     plt.show()

# metric_idx = 2  # Jerk
# robot_labels = [r.upper() for r in robots]
# width = 0.25

# for jerk_type, joint_name, ylabel, unit in [
#     ("Translational", "EEF_Cartesian", "EEF Translational RMS Jerk", "[m/s³]"),
#     ("Angular", "EEF_Angular", "EEF Angular RMS Jerk", "[deg/s³]"),
# ]:
#     plt.figure(figsize=(8, 5))
#     for i, planner in enumerate(planners):
#         jerk_vals = []
#         for robot in robots:
#             eef_stats = all_rms_stats[robot][planner].get(joint_name, [])
#             vals = [v[metric_idx] for v in eef_stats if v is not None]
#             mean_jerk = np.mean(vals) if vals else np.nan
#             jerk_vals.append(mean_jerk)
#         # Offset x so bars don't overlap
#         plt.bar(
#             np.arange(len(robots)) + i*width,
#             jerk_vals,
#             width,
#             label=planner_labels[planner]
#         )
#     plt.xticks(np.arange(len(robots)) + width, robot_labels)
#     plt.ylabel(f"{ylabel} {unit}")
#     plt.yscale("log")
#     plt.title(f"{jerk_type} End-Effector Jerk (All Robots, All Planners)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# metric_vel_idx = 0  # Index for velocity
# metric_jerk_idx = 2 # Index for jerk

# for planner in planners:
#     cart_scores = []
#     ang_scores = []
#     for robot in robots:
#         # Cartesian
#         cart = all_rms_stats[robot][planner].get("EEF_Cartesian", [])
#         ang = all_rms_stats[robot][planner].get("EEF_Angular", [])

#         if cart:
#             vel_vals = [v[metric_vel_idx] for v in cart if v is not None]
#             jerk_vals = [v[metric_jerk_idx] for v in cart if v is not None]
#             # Compute mean velocity and mean jerk for this robot/planner
#             mean_vel = np.mean(vel_vals) if vel_vals else np.nan
#             mean_jerk = np.mean(jerk_vals) if jerk_vals else np.nan
#             score = mean_jerk / mean_vel if mean_vel > 1e-8 else np.nan
#             cart_scores.append(score)
#         else:
#             cart_scores.append(np.nan)

#         # Angular
#         if ang:
#             vel_vals_ang = [v[metric_vel_idx] for v in ang if v is not None]
#             jerk_vals_ang = [v[metric_jerk_idx] for v in ang if v is not None]
#             mean_vel_ang = np.mean(vel_vals_ang) if vel_vals_ang else np.nan
#             mean_jerk_ang = np.mean(jerk_vals_ang) if jerk_vals_ang else np.nan
#             score_ang = mean_jerk_ang / mean_vel_ang if mean_vel_ang > 1e-8 else np.nan
#             ang_scores.append(score_ang)
#         else:
#             ang_scores.append(np.nan)

#     x = np.arange(len(robots))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(8, 4))
#     b1 = ax.bar(x - width/2, cart_scores, width, label="EEF Cartesian (Jerk / Velocity)")
#     b2 = ax.bar(x + width/2, ang_scores, width, label="EEF Angular (Jerk / Velocity)")
#     ax.set_xticks(x)
#     ax.set_xticklabels([r.upper() for r in robots])
#     ax.set_ylabel("Velocity-adjusted RMS Jerk\n[$s^{-1}$] (Lower = Smoother & Faster)")
#     # ax.set_yscale("log")
#     ax.set_title(f"{planner_labels[planner]}: Velocity-adjusted EEF RMS Jerk")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

# for planner in planners:
#     for m_idx, metric in enumerate(metrics):
#         cart_vals = []
#         ang_vals = []
#         for robot in robots:
#             # Get means over all runs for this robot, planner, and metric
#             cart = all_rms_stats[robot][planner].get("EEF_Cartesian", [])
#             ang = all_rms_stats[robot][planner].get("EEF_Angular", [])
#             cart_val = np.nan
#             ang_val = np.nan
#             if cart:
#                 metric_cart_vals = [v[m_idx] for v in cart if v is not None]
#                 if metric_cart_vals:
#                     cart_val = np.mean(metric_cart_vals)
#             if ang:
#                 metric_ang_vals = [v[m_idx] for v in ang if v is not None]
#                 if metric_ang_vals:
#                     ang_val = np.mean(metric_ang_vals)
#             cart_vals.append(cart_val)
#             ang_vals.append(ang_val)

#         x = np.arange(len(robots))
#         width = 0.35

#         fig, ax = plt.subplots(figsize=(8, 4))
#         b1 = ax.bar(x - width/2, cart_vals, width, label="EEF Cartesian")
#         b2 = ax.bar(x + width/2, ang_vals, width, label="EEF Angular")
#         ax.set_xticks(x)
#         ax.set_xticklabels([r.upper() for r in robots])
#         ax.set_ylabel(f"{metrics[m_idx]} {metric_units_cart[m_idx]}")
#         ax.set_title(f"{planner_labels[planner]}: EEF {metrics[m_idx]} (All Robots)")
#         ax.set_yscale("log")
#         ax.legend()
#         plt.tight_layout()
#         plt.show()

planner_colors = {
    "cart": "#8D5A99",   # Slate blue
    "pilz": "#e76f51",   # Burnt orange/burgundy
    "servo": "#2186a5"   # Deep purple
}

metrics = ['vel', 'acc', 'jerk']
metric_names = ['Velocity (RMS)', 'Acceleration (RMS)', 'Jerk (RMS)']
# metric_units = ['[m/s]', '[m/s²]', '[m/s³]']
metric_units = ['[rad/s]', '[rad/s²]', '[rad/s³]']

for planner in planners:
    # Collect max number of physical joints (ignore EEF entries)
    max_joints = max(
        sum(1 for j in all_rms_stats[r][planner].keys()
            if j not in ("EEF_Cartesian", "EEF_Angular"))
        for r in robots
    )
    joint_labels = [f"J{i+1}" for i in range(max_joints)]

    fig, axs = plt.subplots(3, 1, figsize=(max(8, max_joints*1.3), 9), sharex=True)
    for metric_idx in range(3):  # 0: vel, 1: acc, 2: jerk
        data = np.full((len(robots), max_joints), np.nan)
        for r_idx, robot in enumerate(robots):
            phys_joints = [j for j in sorted(all_rms_stats[robot][planner].keys())
                           if j not in ("EEF_Cartesian", "EEF_Angular")]
            for j_idx, joint in enumerate(phys_joints):
                vals = all_rms_stats[robot][planner][joint]
                metric_vals = [v[metric_idx] for v in vals if v is not None]
                if metric_vals:
                    data[r_idx, j_idx] = np.mean(metric_vals)
        im = axs[metric_idx].imshow(data, aspect='auto', cmap='viridis')
        axs[metric_idx].set_yticks(np.arange(len(robots)))
        axs[metric_idx].set_yticklabels([r.upper() for r in robots])
        axs[metric_idx].set_xticks(np.arange(max_joints))
        axs[metric_idx].set_xticklabels(joint_labels, rotation=30)
        axs[metric_idx].set_ylabel("Robot")
        axs[metric_idx].set_title(f"{metric_names[metric_idx]}")
        cbar = plt.colorbar(im, ax=axs[metric_idx], shrink=0.9)
        cbar.set_label(metric_units[metric_idx])
    axs[-1].set_xlabel("Joint")
    fig.suptitle(f"{planner_labels[planner]}: RMS Velocity, Acceleration, Jerk per Joint (All Robots)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()










velocity_consistency_stats = {robot: {planner: {} for planner in planners} for robot in robots}

for robot in robots:
    for planner in planners:
        data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
        eef_files = sorted(glob.glob(os.path.join(data_dir, "eef*")))
        eef_cvs = []
        for ef in eef_files:
            df = pd.read_csv(ef)
            t = df["time"].values
            # Assume columns are ['x', 'y', 'z']
            pos = df[["x", "y", "z"]].values
            if len(pos) < window_length: continue
            dt = np.mean(np.diff(t))
            vel = savgol_filter(pos, window_length, polyorder, deriv=1, delta=dt, mode="interp", axis=0)
            vel_mag = np.linalg.norm(vel, axis=1)
            if np.mean(vel_mag) > 1e-8:
                cv = np.std(vel_mag) / np.mean(vel_mag)
                vel_cons = -np.log10(cv)
                # vel_cons = (cv)
                eef_cvs.append(vel_cons)
                if robot == "kuka":  # or whichever is problematic
                    print(f"KUKA {planner} run: mean={np.mean(vel_mag):.4e}, std={np.std(vel_mag):.4e}, CV={cv:.4e}, VelCons={vel_cons:.4f}")
        if eef_cvs:
            velocity_consistency_stats[robot][planner]["EEF_Cartesian"] = np.mean(eef_cvs)
        else:
            velocity_consistency_stats[robot][planner]["EEF_Cartesian"] = np.nan

# for robot in robots:
#     for planner in planners:
#         # Joint velocity CVs (across all joints)
#         # EEF_Cartesian velocity CV
#         eef_vals = all_rms_stats[robot][planner].get("EEF_Cartesian", [])
#         eef_vels = [v[0] for v in eef_vals if v is not None]
#         if eef_vels and np.mean(eef_vels) > 1e-8:
#             velocity_consistency_stats[robot][planner]["EEF_Cartesian"] = -np.log10(np.std(eef_vels) / np.mean(eef_vels))
#             # velocity_consistency_stats[robot][planner]["EEF_Cartesian"] = np.std(eef_vels) / np.mean(eef_vels)
#         else:
#             velocity_consistency_stats[robot][planner]["EEF_Cartesian"] = np.nan

# --- Make LaTeX Table ---
velocity_cv_table = []
for planner in planners:
    row = [planner_labels[planner]]
    for robot in robots:
        # EEF CV
        eef_cv = velocity_consistency_stats[robot][planner].get("EEF_Cartesian", np.nan)
        # Mean joint CV across all joints
        joint_cvs = [
            cv for joint, cv in velocity_consistency_stats[robot][planner].items()
            if not joint.startswith("EEF") and not np.isnan(cv)
        ]
        mean_joint_cv = np.nanmean(joint_cvs) if joint_cvs else np.nan
        row.append(f"{eef_cv:.3f}")
        row.append(f"{mean_joint_cv:.3f}")
    velocity_cv_table.append(row)

headers = [
    "Planner",
    "UR20 EEF", "UR20 Joint",
    "FANUC EEF", "FANUC Joint",
    "UR5 EEF", "UR5 Joint",
    "KUKA EEF", "KUKA Joint"
]
print(tabulate(velocity_cv_table, headers=headers, tablefmt="latex"))

# --- Bar Plot: EEF CV per robot/planner ---
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.22
x = np.arange(len(robots))
for i, planner in enumerate(planners):
    eef_cvs = []
    for robot in robots:
        cv = velocity_consistency_stats[robot][planner].get("EEF_Cartesian", np.nan)
        eef_cvs.append(cv)
    ax.bar(x + i*width, eef_cvs, width, label=planner_labels[planner], color=planner_colors[planner], edgecolor="black", linewidth=0.7)
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, color='gray', alpha=0.8)
ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.7, color='gray', alpha=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels([r.upper() for r in robots])
ax.set_ylabel("Velocity Consistency (CV, EEF)")
ax.set_title("EEF Velocity Consistency (Coefficient of Variation, Lower = More Consistent)")
ax.set_yscale("log")  # <- This sets log scale!
ax.set_axisbelow(True)

ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.22
n_robots = len(robots)
x = np.arange(n_robots + 1)  # robots plus "Average"

for i, planner in enumerate(planners):
    # Collect EEF CVs for all robots
    eef_cvs = []
    for robot in robots:
        cv = velocity_consistency_stats[robot][planner].get("EEF_Cartesian", np.nan)
        eef_cvs.append(cv)
    avg_cv = np.nanmean(eef_cvs)
    all_cvs = eef_cvs + [avg_cv]  # Add average as fifth bar

    # For correct bar alignment, shift each planner group
    ax.bar(x + i*width, all_cvs, width,
           label=planner_labels[planner],
           color=planner_colors[planner],
           edgecolor="black", linewidth=0.7)

# Set ticks in the middle of grouped bars
ax.set_xticks(x + width)
ax.set_xticklabels([r.upper() for r in robots] + ["Average"])
ax.set_ylabel("Velocity Consistency")
ax.set_title("EEF Translational Velocity Consistency (Coefficient of Variation, Higher = More Consistent)", pad=35)
# ax.set_yscale("log")
ax.set_axisbelow(True)
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, color='gray', alpha=0.8)
ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.7, color='gray', alpha=0.8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=True)

plt.tight_layout()
plt.show()


velocity_cv_table = []
for i, planner in enumerate(planners):
    row = [planner_labels[planner]]
    eef_cvs = []
    for robot in robots:
        cv = velocity_consistency_stats[robot][planner].get("EEF_Cartesian", np.nan)
        eef_cvs.append(cv)
        row.append(f"{cv:.3f}" if not np.isnan(cv) else "N/A")
    avg_cv = np.nanmean(eef_cvs)
    row.append(f"{avg_cv:.3f}" if not np.isnan(avg_cv) else "N/A")
    velocity_cv_table.append(row)

headers = ["Planner"] + [f"{r.upper()} " for r in robots] + ["Average"]

print(tabulate(velocity_cv_table, headers=headers, tablefmt="latex"))

# # robots and planners lists already defined

# for i, metric in enumerate(metrics):
#     # Build data matrix: rows=robots, cols=planners
#     data = np.zeros((len(robots), len(planners)))
#     for r_idx, robot in enumerate(robots):
#         for p_idx, planner in enumerate(planners):
#             vals = []
#             # Aggregate all joints for robot/planner/metric
#             for joint_vals in all_rms_stats[robot][planner].values():
#                 vals.extend([v[i] for v in joint_vals if v is not None])
#             data[r_idx, p_idx] = np.mean(vals) if vals else np.nan

#     plt.figure(figsize=(7, 4))
#     im = plt.imshow(data, aspect='auto', cmap='viridis')
#     plt.xticks(np.arange(len(planners)), [planner_labels[p] for p in planners])
#     plt.yticks(np.arange(len(robots)), [r.upper() for r in robots])
#     plt.colorbar(im, label=metric_units[i])
#     plt.title(f"{metric_names[i]} (RMS) for Each Robot & Planner")
#     plt.xlabel("Motion Planner")
#     plt.ylabel("Robot")
#     plt.tight_layout()
#     plt.show()


# metrics = ["Velocity (RMS)", "Acceleration (RMS)", "Jerk (RMS)"]

# for planner in planners:
#     n_robots = len(robots)
#     nrows, ncols = 2, 2
#     fig, axs = plt.subplots(nrows, ncols, figsize=(12, 9), sharey=False)
#     axs = axs.flatten()  # Makes it easier to index

#     for r_idx, robot in enumerate(robots):
#         ax = axs[r_idx]
#         joint_names = [j for j in sorted(all_rms_stats[robot][planner].keys())
#                        if j not in ("EEF_Cartesian", "EEF_Angular")]
#         njoints = len(joint_names)
#         data = np.zeros((njoints, len(metrics)))
#         for j_idx, joint in enumerate(joint_names):
#             vals = all_rms_stats[robot][planner].get(joint, [])
#             if vals:
#                 for m_idx in range(len(metrics)):
#                     metric_vals = [v[m_idx] for v in vals if v is not None]
#                     data[j_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#             else:
#                 data[j_idx, :] = np.nan
#         im = ax.imshow(data, aspect="auto", cmap="viridis")
#         ax.set_xticks(np.arange(len(metrics)))
#         ax.set_xticklabels(metrics, rotation=25)
#         ax.set_yticks(np.arange(njoints))
#         ax.set_yticklabels([f"J{i+1}" for i in range(njoints)])
        
#         ax.set_title(robot.upper())
#         fig.colorbar(im, ax=ax, shrink=0.7)
#     # Hide any unused subplots (if robots < 4)
#     for i in range(len(robots), nrows*ncols):
#         fig.delaxes(axs[i])
#     fig.suptitle(f"{planner_labels[planner]}: RMS Velocity, Acceleration, Jerk per Joint (All Robots)", fontsize=20)
#     fig.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()

# # --- 1. Physical joints: one plot, all robots as subplots, for each planner ---
# for planner in planners:
#     fig, axs = plt.subplots(1, len(robots), figsize=(6*len(robots), 8), sharey=False)
#     if len(robots) == 1:
#         axs = [axs]
#     for r_idx, robot in enumerate(robots):
#         # Only physical joints
#         joint_names = [j for j in sorted(all_rms_stats[robot][planner].keys())
#                        if j not in ("EEF_Cartesian", "EEF_Angular")]
#         njoints = len(joint_names)
#         data = np.zeros((njoints, len(metrics)))
#         for j_idx, joint in enumerate(joint_names):
#             vals = all_rms_stats[robot][planner].get(joint, [])
#             if vals:
#                 for m_idx in range(len(metrics)):
#                     metric_vals = [v[m_idx] for v in vals if v is not None]
#                     data[j_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#             else:
#                 data[j_idx, :] = np.nan
#         im = axs[r_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[r_idx].set_xticks(np.arange(len(metrics)))
#         axs[r_idx].set_xticklabels(metrics, rotation=25)
#         axs[r_idx].set_yticks(np.arange(njoints))
#         axs[r_idx].set_yticklabels(joint_names)
#         axs[r_idx].set_title(robot.upper())
#         fig.colorbar(im, ax=axs[r_idx], shrink=0.7)
#     fig.suptitle(f"{planner_labels[planner]}: RMS Velocity, Acceleration, Jerk per Joint (All Robots)", fontsize=20)
#     fig.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()

# --- 2. EEF_Cartesian and EEF_Angular: each gets its own plot, robots as rows, planners as columns ---
# eef_types = ["EEF_Cartesian", "EEF_Angular"]

# for eef_type in eef_types:
#     fig, axs = plt.subplots(1, len(robots), figsize=(6*len(robots), 6), sharey=True)
#     if len(robots) == 1:
#         axs = [axs]
#     for r_idx, robot in enumerate(robots):
#         data = np.full((len(planners), len(metrics)), np.nan)
#         for p_idx, planner in enumerate(planners):
#             vals = all_rms_stats[robot][planner].get(eef_type, [])
#             if vals:
#                 for m_idx in range(len(metrics)):
#                     metric_vals = [v[m_idx] for v in vals if v is not None]
#                     data[p_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#         im = axs[r_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[r_idx].set_xticks(np.arange(len(metrics)))
#         axs[r_idx].set_xticklabels(metrics, rotation=25)
#         axs[r_idx].set_yticks(np.arange(len(planners)))
#         axs[r_idx].set_yticklabels([planner_labels[p] for p in planners])
#         axs[r_idx].set_title(robot.upper())
#         fig.colorbar(im, ax=axs[r_idx], shrink=0.7)
#     fig.suptitle(f"{eef_type}: RMS Velocity, Acceleration, Jerk (All Robots, All Planners)", fontsize=20)
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# for planner in planners:
#     data = np.full((len(robots), len(metrics)), np.nan)
#     for r_idx, robot in enumerate(robots):
#         vals = all_rms_stats[robot][planner].get(eef_type, [])
#         if vals:
#             for m_idx in range(len(metrics)):
#                 metric_vals = [v[m_idx] for v in vals if v is not None]
#                 data[r_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#     plt.figure(figsize=(7, 4))
#     im = plt.imshow(data, aspect="auto", cmap="viridis")
#     plt.xticks(np.arange(len(metrics)), metrics, rotation=25)
#     plt.yticks(np.arange(len(robots)), [r.upper() for r in robots])
#     plt.colorbar(im, shrink=0.8)
#     plt.title(f"{planner_labels[planner]}: {eef_type} RMS Velocity, Acceleration, Jerk (All Robots)")
#     plt.tight_layout()
#     plt.show()


# # Plots combined results for all robots
# eef_types = ["EEF_Cartesian", "EEF_Angular"]
# for eef_type in eef_types:
#     fig, axs = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6), sharey=True)
#     for m_idx, metric in enumerate(metrics):
#         data = np.full((len(robots), len(planners)), np.nan)
#         for r_idx, robot in enumerate(robots):
#             for p_idx, planner in enumerate(planners):
#                 vals = all_rms_stats[robot][planner].get(eef_type, [])
#                 metric_vals = [v[m_idx] for v in vals if v is not None]
#                 if metric_vals:
#                     data[r_idx, p_idx] = np.mean(metric_vals)
#         im = axs[m_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[m_idx].set_xticks(np.arange(len(planners)))
#         axs[m_idx].set_xticklabels([planner_labels[p] for p in planners])
#         axs[m_idx].set_yticks(np.arange(len(robots)))
#         axs[m_idx].set_yticklabels([r.upper() for r in robots])
#         axs[m_idx].set_title(f"{metric}")
#         fig.colorbar(im, ax=axs[m_idx], shrink=0.7)
#     fig.suptitle(f"{eef_type}: RMS Velocity, Acceleration, Jerk (All Robots, All Planners)", fontsize=20)
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# metrics = ["Velocity (RMS)", "Acceleration (RMS)", "Jerk (RMS)"]

# for planner in planners:
#     n_robots = len(robots)
#     fig, axs = plt.subplots(1, n_robots, figsize=(5*n_robots, 7), sharey=False)
#     if n_robots == 1:
#         axs = [axs]
#     for r_idx, robot in enumerate(robots):
#         # Collect joint names, including "EEF_Cartesian" if present
#         joint_names = sorted(all_rms_stats[robot][planner].keys())
#         # Move EEF_Cartesian to the end if present
#         if "EEF_Cartesian" in joint_names:
#             joint_names.remove("EEF_Cartesian")
#             joint_names.append("EEF_Cartesian")
#         njoints = len(joint_names)
#         data = np.zeros((njoints, len(metrics)))
#         for j_idx, joint in enumerate(joint_names):
#             vals = all_rms_stats[robot][planner].get(joint, [])
#             if vals:
#                 for m_idx in range(len(metrics)):
#                     metric_vals = [v[m_idx] for v in vals if v is not None]
#                     data[j_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#             else:
#                 data[j_idx, :] = np.nan
#         im = axs[r_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[r_idx].set_xticks(np.arange(len(metrics)))
#         axs[r_idx].set_xticklabels(metrics, rotation=25)
#         axs[r_idx].set_yticks(np.arange(njoints))
#         axs[r_idx].set_yticklabels(joint_names)
#         axs[r_idx].set_title(f"{robot.upper()}")
#         fig.colorbar(im, ax=axs[r_idx], shrink=0.7)
#     fig.suptitle(f"{planner_labels[planner]}: RMS Velocity, Acceleration, Jerk per Joint (All Robots)\n(EEF_Cartesian included)", fontsize=18)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# for which in ["EEF_Cartesian", "EEF_Angular"]:
#     fig, axs = plt.subplots(1, len(robots), figsize=(5*len(robots), 7))
#     if len(robots) == 1:
#         axs = [axs]
#     for r_idx, robot in enumerate(robots):
#         data = []
#         for planner in planners:
#             vals = all_rms_stats[robot][planner].get(which, [])
#             if vals:
#                 data.append(np.mean(vals, axis=0))  # mean over runs
#             else:
#                 data.append([np.nan, np.nan, np.nan])
#         data = np.array(data).T  # Shape: (metric, planner)
#         im = axs[r_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[r_idx].set_xticks(np.arange(len(planners)))
#         axs[r_idx].set_xticklabels([planner_labels[p] for p in planners])
#         axs[r_idx].set_yticks(np.arange(3))
#         axs[r_idx].set_yticklabels(["RMS Velocity", "RMS Acceleration", "RMS Jerk"])
#         axs[r_idx].set_title(f"{robot.upper()}")
#         fig.colorbar(im, ax=axs[r_idx], shrink=0.7)
#     fig.suptitle(f"{which}: RMS Velocity, Acceleration, Jerk (All Robots)", fontsize=18)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# metrics = ["Velocity (RMS)", "Acceleration (RMS)", "Jerk (RMS)"]

# for planner in planners:
#     n_robots = len(robots)
#     fig, axs = plt.subplots(1, n_robots, figsize=(5*n_robots, 7), sharey=False)
#     if n_robots == 1:
#         axs = [axs]
#     for r_idx, robot in enumerate(robots):
#         joint_names = sorted(all_rms_stats[robot][planner].keys())
#         njoints = len(joint_names)
#         data = np.zeros((njoints, len(metrics)))
#         for j_idx, joint in enumerate(joint_names):
#             vals = all_rms_stats[robot][planner].get(joint, [])
#             if vals:
#                 for m_idx in range(len(metrics)):
#                     metric_vals = [v[m_idx] for v in vals if v is not None]
#                     data[j_idx, m_idx] = np.mean(metric_vals) if metric_vals else np.nan
#             else:
#                 data[j_idx, :] = np.nan
#         im = axs[r_idx].imshow(data, aspect="auto", cmap="viridis")
#         axs[r_idx].set_xticks(np.arange(len(metrics)))
#         axs[r_idx].set_xticklabels(metrics, rotation=25)
#         axs[r_idx].set_yticks(np.arange(njoints))
#         axs[r_idx].set_yticklabels(joint_names)
#         axs[r_idx].set_title(f"{robot.upper()}")
#         fig.colorbar(im, ax=axs[r_idx], shrink=0.7)
#     fig.suptitle(f"{planner_labels[planner]}: RMS Velocity, Acceleration, Jerk per Joint (All Robots)", fontsize=18)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# # For each robot, plot a heatmap of mean RMS (velocity/acc/jerk) per joint, per planner
# for robot in robots:
#     joint_names = sorted(set(j for p in planners for j in all_rms_stats[robot][p].keys()))
#     njoints = len(joint_names)
#     fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
#     for pi, metric in enumerate(["Velocity", "Acceleration", "Jerk"]):
#         data = np.zeros((njoints, len(planners)))
#         for j_idx, joint in enumerate(joint_names):
#             for p_idx, planner in enumerate(planners):
#                 # List of (rms_vel, rms_acc, rms_jerk) for this joint+planner+robot, for all runs
#                 vals = all_rms_stats[robot][planner].get(joint, [])
#                 if vals:
#                     metric_vals = [v[pi] for v in vals if v is not None]
#                     data[j_idx, p_idx] = np.mean(metric_vals) if metric_vals else np.nan
#                 else:
#                     data[j_idx, p_idx] = np.nan
#         im = axs[pi].imshow(data, aspect="auto", cmap="viridis")
#         axs[pi].set_xticks(np.arange(len(planners)))
#         axs[pi].set_xticklabels([planner_labels[p] for p in planners], rotation=25)
#         axs[pi].set_yticks(np.arange(njoints))
#         if pi == 0:
#             axs[pi].set_yticklabels(joint_names)
#         else:
#             axs[pi].set_yticklabels([])
#         axs[pi].set_title(f"{metric} (RMS)")

#         fig.colorbar(im, ax=axs[pi], shrink=0.7)
#     fig.suptitle(f"{robot.upper()} - Joint RMS (Velocity / Acceleration / Jerk)", fontsize=16)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# # --- Optionally: Print mean RMS table per robot/planner/metric ---
# for robot in robots:
#     print(f"\n==== {robot.upper()} ====")
#     joint_names = sorted(set(j for p in planners for j in all_rms_stats[robot][p].keys()))
#     for pi, metric in enumerate(["Velocity", "Acceleration", "Jerk"]):
#         print(f"\n--- {metric} (RMS) ---")
#         print("Joint".ljust(20), end="")
#         for p in planners:
#             print(f"{planner_labels[p]:>15}", end="")
#         print()
#         for j in joint_names:
#             print(j.ljust(20), end="")
#             for p in planners:
#                 vals = all_rms_stats[robot][p].get(j, [])
#                 metric_vals = [v[pi] for v in vals if v is not None]
#                 v = np.mean(metric_vals) if metric_vals else np.nan
#                 print(f"{v:15.3e}", end="")
#             print()


# pooled_rms = {planner: {'vel': [], 'acc': [], 'jerk': []} for planner in planners}

# for robot in robots:
#     for planner in planners:
#         for joint, vals in all_rms_stats[robot][planner].items():
#             for v in vals:
#                 if v is not None:
#                     pooled_rms[planner]['vel'].append(v[0])
#                     pooled_rms[planner]['acc'].append(v[1])
#                     pooled_rms[planner]['jerk'].append(v[2])

# # Example: boxplot for RMS velocity
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# for i, metric in enumerate(['vel', 'acc', 'jerk']):
#     data = [pooled_rms[planner][metric] for planner in planners]
#     axs[i].boxplot(data, labels=[planner_labels[p] for p in planners], showmeans=True)
#     axs[i].set_title(f'RMS {metric.capitalize()}')
#     axs[i].set_ylabel(f'RMS {metric.capitalize()}')
# fig.suptitle("Pooled RMS Metrics Per Planner (All Robots & Joints)", fontsize=16)
# plt.tight_layout()
# plt.show()

# metric_idx = 2  # 0: velocity, 1: acc, 2: jerk
# metric_name = ["Velocity", "Acceleration", "Jerk"][metric_idx]

# joint_names = sorted(set(j for robot in robots for planner in planners for j in all_rms_stats[robot][planner].keys()))

# for planner in planners:
#     data = np.zeros((len(joint_names), len(robots)))
#     for j_idx, joint in enumerate(joint_names):
#         for r_idx, robot in enumerate(robots):
#             vals = [v[metric_idx] for v in all_rms_stats[robot][planner].get(joint, []) if v is not None]
#             data[j_idx, r_idx] = np.mean(vals) if vals else np.nan

#     fig, ax = plt.subplots(figsize=(1.2*len(robots), 0.5*len(joint_names)+2))
#     im = ax.imshow(data, aspect='auto', cmap='viridis')
#     ax.set_xticks(np.arange(len(robots)))
#     ax.set_xticklabels([r.upper() for r in robots])
#     ax.set_yticks(np.arange(len(joint_names)))
#     ax.set_yticklabels(joint_names, fontsize=9)
#     ax.set_title(f'{planner_labels[planner]}: RMS {metric_name} per Joint/Robot', pad=12)
#     plt.colorbar(im, ax=ax, shrink=0.8, label=f'RMS {metric_name}')
#     fig.tight_layout()
#     plt.show()

# data = np.zeros((len(joint_names), len(planners)))
# for j_idx, joint in enumerate(joint_names):
#     for p_idx, planner in enumerate(planners):
#         vals = []
#         for robot in robots:
#             vals += [v[metric_idx] for v in all_rms_stats[robot][planner].get(joint, []) if v is not None]
#         data[j_idx, p_idx] = np.mean(vals) if vals else np.nan

# fig, ax = plt.subplots(figsize=(1.1*len(planners)+2, 0.5*len(joint_names)+2))
# im = ax.imshow(data, aspect='auto', cmap='viridis')
# ax.set_xticks(np.arange(len(planners)))
# ax.set_xticklabels([planner_labels[p] for p in planners])
# ax.set_yticks(np.arange(len(joint_names)))
# ax.set_yticklabels(joint_names, fontsize=9)
# ax.set_title(f'RMS {metric_name} per Joint/Planner (Pooled over Robots)', pad=12)
# plt.colorbar(im, ax=ax, shrink=0.8, label=f'RMS {metric_name}')
# fig.tight_layout()
# plt.show()

