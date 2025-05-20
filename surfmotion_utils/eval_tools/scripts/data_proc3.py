import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ament_index_python.packages import get_package_share_directory
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import json
from matplotlib.ticker import EngFormatter, FuncFormatter

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

robots = ["ur20", "fanuc", "ur5", "kuka"]
planners = ["cart", "pilz", "servo"]
planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
# planner_colors = {"cart": "#1f77b4", "pilz": "#ff7f0e", "servo": "#2ca02c"}

# planner_colors = {
#     "cart": "#264653",  # Slate blue
#     "pilz": "#e76f51",  # Burnt orange/burgundy
#     "servo": "#2186a5"  # Light teal
# }

planner_colors = {
    "cart": "#8D5A99",   # Slate blue
    "pilz": "#e76f51",   # Burnt orange/burgundy
    "servo": "#2186a5"   # Deep purple
}

mean_pos_err = np.zeros((len(robots), len(planners)))
mean_ang_err = np.zeros((len(robots), len(planners)))
max_pos_err = np.zeros((len(robots), len(planners)))
max_ang_err = np.zeros((len(robots), len(planners)))
rms_pos_err = np.zeros((len(robots), len(planners)))
rms_ang_err = np.zeros((len(robots), len(planners)))
rms_pos_err_std = np.zeros((len(robots), len(planners)))
rms_ang_err_std =  np.zeros((len(robots), len(planners)))


kuka_offset = np.array([0.4, 0, 0]) 

for r, robot in enumerate(robots):
    for p, planner in enumerate(planners):
        data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
        eef_files = sorted(glob.glob(os.path.join(data_dir, "eef*")))
        poses_path = os.path.join(data_dir, "poses.json")
        poses = json_load(poses_path)
        ref_positions = np.array([[pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]] for pose in poses])
        ref_orientations = np.array([[pose["orientation"]["x"], pose["orientation"]["y"], pose["orientation"]["z"], pose["orientation"]["w"]] for pose in poses])

        all_pos_errors = []
        all_ang_errors = []

        for eef_file in eef_files:
            df = pd.read_csv(eef_file)
            traj_positions = df[["x", "y", "z"]].values
            traj_orientations = df[["qx", "qy", "qz", "qw"]].values

            if robot == "kuka":
                # Kuka base link is offset from origin
                traj_positions += kuka_offset

            tree = KDTree(traj_positions)
            dists, indices = tree.query(ref_positions)
            ref_R = R.from_quat(ref_orientations)
            traj_R = R.from_quat(traj_orientations[indices])
            rel_rot = ref_R.inv() * traj_R
            ang_errors = rel_rot.magnitude() * (180.0 / np.pi)

            all_pos_errors.append(dists)
            all_ang_errors.append(ang_errors)

        # Stack and compute stats
        all_pos_errors = np.vstack(all_pos_errors)
        all_ang_errors = np.vstack(all_ang_errors)
        mean_pos_err[r, p] = np.mean(all_pos_errors)
        mean_ang_err[r, p] = np.mean(all_ang_errors)
        max_pos_err[r, p] = np.max(all_pos_errors)
        max_ang_err[r, p] = np.max(all_ang_errors)

        # RMS error
        per_run_rms_pos = [np.sqrt(np.mean(d**2)) for d in all_pos_errors]
        per_run_rms_ang = [np.sqrt(np.mean(a**2)) for a in all_ang_errors]
        rms_pos_err[r, p] = np.mean(per_run_rms_pos)
        rms_ang_err[r, p] = np.mean(per_run_rms_ang)

        # Optional: also store std deviation of per-run RMS for error bars
        rms_pos_err_std[r, p] = np.std(per_run_rms_pos)
        rms_ang_err_std[r, p] = np.std(per_run_rms_ang)

fig_width = 8
fig_height = 4.5

scale = 1e6  # m to mm
width = 0.18
x = np.arange(len(robots))
plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams['font.family'] = 'serif'         # or 'Times New Roman'
# plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})

fig_width, fig_height = 7, 4  # use the same values for all plots

# Calculate average over all robots for each planner
avg_rms_pos = np.nanmean(rms_pos_err, axis=0)  # shape: (3,)
avg_rms_ang = np.nanmean(rms_ang_err, axis=0)  # shape: (3,)

# Append to arrays (shape: (5, 3) after)
rms_pos_err_ext = np.vstack([rms_pos_err, avg_rms_pos[None, :]])
rms_ang_err_ext = np.vstack([rms_ang_err, avg_rms_ang[None, :]])

robots_ext = robots + ["avg"]
x_ext = np.arange(len(robots_ext))

def custom_um_mm_formatter(y, pos):
    if y < 1000:
        return f"{y:.0f} µm"
    else:
        return f"{y/1000:.2f} mm"

def deg_formatter(y, pos):
    if y >= 1:
        return f"{y:.0f}°"
    elif y >= 0.1:
        return f"{y:.2f}°"
    else:
        return f"{y:.3f}°"

# --- Mean Position Error Plot ---
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for p, planner in enumerate(planners):
    ax.bar(
        x_ext + (p-1)*width, rms_pos_err_ext[:, p]*scale, width,
        label=planner_labels[planner],
        color=planner_colors[planner],
        edgecolor="black", linewidth=0.7
    )

ax.set_xticks(x_ext)
ax.set_xticklabels([r.upper() for r in robots] + ["Average"])
ax.set_ylabel("Mean RMS Position Error")
ax.set_title("Mean RMS Position Error per Robot & Planner", pad=30)
ax.set_yscale("log")
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(FuncFormatter(custom_um_mm_formatter))

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True)
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, color='gray', alpha=0.8)
ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.7, color='gray', alpha=0.8)

plt.tight_layout()
plt.show()

# --- Mean Angular Error Plot ---
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for p, planner in enumerate(planners):
    ax.bar(
        x_ext + (p-1)*width, rms_ang_err_ext[:, p], width,
        label=planner_labels[planner],
        color=planner_colors[planner],
        edgecolor="black", linewidth=0.7
    )

ax.set_xticks(x_ext)
ax.set_xticklabels([r.upper() for r in robots] + ["Average"])
ax.set_ylabel("Mean RMS Angular Error [deg]")
ax.set_title("Mean RMS Angular Error per Robot & Planner", pad=30)
ax.set_yscale("log")
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(FuncFormatter(deg_formatter))

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True)
ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.8, color='gray', alpha=0.8)
ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.7, color='gray', alpha=0.8)

plt.tight_layout()
plt.show()

# --- Print table ---
print("\nRobot".ljust(8), end="")
for planner in planners:
    print(f"{planner_labels[planner]:>18}", end="")
print()
for r, robot in enumerate(robots):
    print(f"{robot.upper():<8}", end="")
    for p, planner in enumerate(planners):
        print(f" {mean_pos_err[r, p]:.3e} m, {mean_ang_err[r, p]:.2f} deg", end="")
    print()

print("\nRobot".ljust(8), end="")
for planner in planners:
    print(f"{planner_labels[planner]:>32}", end="")  # wider columns for both errors
print()
print("".ljust(8), end="")
for planner in planners:
    print(f"{'Mean / RMS [m] | Mean / RMS [deg]':>32}", end="")
print()

for r, robot in enumerate(robots):
    print(f"{robot.upper():<8}", end="")
    for p, planner in enumerate(planners):
        print(f" {mean_pos_err[r, p]:.2e}/{rms_pos_err[r, p]:.2e} m | {mean_ang_err[r, p]:.2f}/{rms_ang_err[r, p]:.2f} deg".rjust(32), end="")
    print()

from tabulate import tabulate

headers = ["Robot"] + [f"{planner_labels[p]}\nPos. [m]" for p in planners]
table = []

for r, robot in enumerate(robots):
    row = [robot.upper()]
    # Add position errors
    for p in range(len(planners)):
        row.append(f"{mean_pos_err[r, p]:.2e}")
    table.append(row)

latex = tabulate(table, headers=headers, tablefmt="latex")
print("\nLaTeX table:\n")
print(latex)



headers = ["Robot"] + [f"{planner_labels[p]}\nAng. [deg]" for p in planners]
table = []

for r, robot in enumerate(robots):
    row = [robot.upper()]
    # Add angular errors
    for p in range(len(planners)):
        row.append(f"{mean_ang_err[r, p]:.2f}")
    table.append(row)

latex = tabulate(table, headers=headers, tablefmt="latex")
print("\nLaTeX table:\n")
print(latex)


headers = ["Robot"]
# First position error columns for all planners
for p in planners:
    headers.append(f"{planner_labels[p]} [m]")
# Then angular error columns for all planners
for p in planners:
    headers.append(f"{planner_labels[p]} [deg]")

table = []
for r, robot in enumerate(robots):
    row = [robot.upper()]
    # Add position errors for all planners
    for p in range(len(planners)):
        row.append(f"{rms_pos_err[r, p]:.2e}")
    # Add angular errors for all planners
    for p in range(len(planners)):
        row.append(f"{rms_ang_err[r, p]:.2f}")
    table.append(row)

latex = tabulate(table, headers=headers, tablefmt="latex")
print("\nLaTeX table:\n")
print(latex)