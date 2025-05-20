import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import json

# === User setup ===
robots = ["ur20", "fanuc", "ur5", "kuka"]
planners = ["cart", "pilz", "servo"]
planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
planner_colors = {
    "cart": "#8D5A99",   # Slate blue
    "pilz": "#e76f51",   # Burnt orange/burgundy
    "servo": "#2186a5"   # Teal
}

subdirs = {
    "ur20": {
        "cart": "cart_ur20",
        "pilz": "pilz_ur20",
        "servo": "servo_ur20"
    },
    "fanuc": {
        "cart": "cart_fanuc",
        "pilz": "pilz_fanuc",
        "servo": "servo_fanuc"
    },
    "ur5": {
        "cart": "cart_ur5",
        "pilz": "pilz_ur5",
        "servo": "servo_ur5"
    },
    "kuka": {
        "cart": "cart_kuka",
        "pilz": "pilz_kuka",
        "servo": "servo_kuka"
    }
}

def json_load(path):
    with open(path, "r") as f:
        return json.load(f)

# === Summary arrays for mean/max stats ===
mean_vel, max_vel = np.zeros((len(robots), len(planners))), np.zeros((len(robots), len(planners)))
mean_acc, max_acc = np.zeros_like(mean_vel), np.zeros_like(mean_vel)
mean_jerk, max_jerk = np.zeros_like(mean_vel), np.zeros_like(mean_vel)

for r, robot in enumerate(robots):
    for p, planner in enumerate(planners):
        # Path to folder with runs for this robot/planner
        data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
        eef_files = sorted(glob.glob(os.path.join(data_dir, "eef*")))
        color = planner_colors[planner]

        # === Aggregate all runs for mean/max reporting ===
        all_vel = []
        all_acc = []
        all_jerk = []

        for run_idx, eef_file in enumerate(eef_files):
            df = pd.read_csv(eef_file)
            df = df.sort_values("time")
            t = df["time"].values
            # Convert time to seconds if not already (adjust if needed)
            t = t - t[0]

            xyz = df[["x", "y", "z"]].values
            dt = np.diff(t)

            # --- Velocity
            vel = np.diff(xyz, axis=0) / dt[:, None]
            vel_mag = np.linalg.norm(vel, axis=1)

            # --- Acceleration
            acc = np.diff(vel, axis=0) / dt[1:, None]
            acc_mag = np.linalg.norm(acc, axis=1)

            # --- Jerk
            jerk = np.diff(acc, axis=0) / dt[2:, None]
            jerk_mag = np.linalg.norm(jerk, axis=1)

            # Store for aggregate stats
            all_vel.extend(vel_mag)
            all_acc.extend(acc_mag)
            all_jerk.extend(jerk_mag)

            # --- Plot for this run (optional, remove if you only want summary plots)
            fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
            axs[0].plot(t[1:], vel_mag, color=color)
            axs[0].set_ylabel("Velocity [m/s]")
            axs[0].set_title(f"{robot.upper()} - {planner_labels[planner]} (Run {run_idx+1})")
            axs[0].grid(True)

            axs[1].plot(t[2:], acc_mag, color=color)
            axs[1].set_ylabel("Acceleration [m/s²]")
            axs[1].grid(True)

            axs[2].plot(t[3:], jerk_mag, color=color)
            axs[2].set_ylabel("Jerk [m/s³]")
            axs[2].set_xlabel("Time [s]")
            axs[2].grid(True)

            plt.tight_layout()
            # plt.show() # Uncomment for interactive use

        # --- Store summary stats for all runs for this robot/planner
        all_vel, all_acc, all_jerk = np.array(all_vel), np.array(all_acc), np.array(all_jerk)
        mean_vel[r, p], max_vel[r, p] = np.mean(all_vel), np.max(all_vel)
        mean_acc[r, p], max_acc[r, p] = np.mean(all_acc), np.max(all_acc)
        mean_jerk[r, p], max_jerk[r, p] = np.mean(all_jerk), np.max(all_jerk)

# === Barplots: Mean/max across robots/planners ===

x = np.arange(len(robots))
width = 0.22

def plot_bar(metric, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, planner in enumerate(planners):
        ax.bar(x + (p-1)*width, metric[:, p], width, label=planner_labels[planner], color=planner_colors[planner])
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in robots])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Plot means
plot_bar(mean_vel, "Mean Velocity [m/s]", "Mean Velocity per Robot & Planner")
plot_bar(mean_acc, "Mean Acceleration [m/s²]", "Mean Acceleration per Robot & Planner")
plot_bar(mean_jerk, "Mean Jerk [m/s³]", "Mean Jerk per Robot & Planner")

# Plot max values (optional)
plot_bar(max_vel, "Max Velocity [m/s]", "Max Velocity per Robot & Planner")
plot_bar(max_acc, "Max Acceleration [m/s²]", "Max Acceleration per Robot & Planner")
plot_bar(max_jerk, "Max Jerk [m/s³]", "Max Jerk per Robot & Planner")

# === Optionally: Print as table for thesis/report ===
from tabulate import tabulate
headers = ["Robot"] + [f"{planner_labels[p]}" for p in planners]
def print_table(arr, label):
    table = []
    for r, robot in enumerate(robots):
        row = [robot.upper()]
        for p in range(len(planners)):
            row.append(f"{arr[r, p]:.3e}")
        table.append(row)
    print(f"\n{label}")
    print(tabulate(table, headers=headers, tablefmt="github"))

print_table(mean_vel, "Mean Velocity [m/s]")
print_table(mean_acc, "Mean Acceleration [m/s²]")
print_table(mean_jerk, "Mean Jerk [m/s³]")

print_table(max_vel, "Max Velocity [m/s]")
print_table(max_acc, "Max Acceleration [m/s²]")
print_table(max_jerk, "Max Jerk [m/s³]")

