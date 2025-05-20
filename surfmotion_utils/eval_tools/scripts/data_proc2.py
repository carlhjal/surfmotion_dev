import os
import glob
from ament_index_python.packages import get_package_share_directory
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import savgol_filter
import json
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def json_load(path):
    with open (path, "r") as f:
        return json.load(f)
    
def plot_polka_line(ax, xs, ys, zs, color1, color2, length=12, label=None, zorder=5):
    """Alternate segments of two colors along a 3D trajectory."""
    n = len(xs)
    toggle = True
    i = 0
    first = True
    while i < n-1:
        next_i = min(i + length, n-1)
        if toggle:
            ax.plot(xs[i:next_i+1], ys[i:next_i+1], zs[i:next_i+1],
                    color=color1, linewidth=2.7, solid_capstyle="round",
                    label=label if first and label else None, zorder=zorder)
        else:
            ax.plot(xs[i:next_i+1], ys[i:next_i+1], zs[i:next_i+1],
                    color=color2, linewidth=2.7, solid_capstyle="round", zorder=zorder)
        first = False
        toggle = not toggle
        i = next_i

# Example usage:


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

robot = "ur20"
planners = ["cart", "pilz", "servo"]

# Prepare colors/markers for planners (optional, for consistency)
# planner_colors = {"cart": "blue", "pilz": "orange", "servo": "green"}
planner_colors = {
    "cart": "#8D5A99",   # Slate blue
    "pilz": "#e76f51",   # Burnt orange/burgundy
    "servo": "#2186a5"   # Deep purple
}
planner_styles = {
    "cart": {"linestyle": "-.",  "marker": "-", "linewidth": 3},   # solid, circles
    "pilz": {"linestyle": ":", "marker": "-", "linewidth": 3},   # dashed, squares
    "servo": {"linestyle": "-", "marker": "-", "linewidth": 3},   # dotted, triangles
}

planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}

# Initialize lists for summary error stats
summary_pos_errors = {}
summary_ang_errors = {}

plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams['font.family'] = 'serif'         # or 'Times New Roman'
# plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
fig_width = 8
fig_height = 4.5

fig2d, ax2d = plt.subplots()
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection="3d")
plt.figure("Position Error per Reference Pose")
plt.figure("Angular Error per Reference Pose")

for planner in planners:
    data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
    eef_files = glob.glob(os.path.join(data_dir, "eef*"))
    poses = json_load(os.path.join(data_dir, "poses.json"))
    ref_positions = np.array([[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in poses])
    ref_orientations = np.array([[p["orientation"]["x"], p["orientation"]["y"], p["orientation"]["z"], p["orientation"]["w"]] for p in poses])
    style = planner_styles[planner]

    # Assume only one run per planner for simplicity; can extend if multiple
    df = pd.read_csv(eef_files[0])
    df["time"] = df["time"] - df["time"].iloc[0]

    # 2D Plot (X vs Y)
    ax2d.plot(df["x"], df["y"], label=planner_labels[planner], color=planner_colors[planner])
    ax2d.scatter(df["x"].iloc[0], df["y"].iloc[0], color=planner_colors[planner], marker="o", s=30)
    ax2d.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color=planner_colors[planner], marker="x", s=30)
    
    N = 10
    x = df["x"]
    y = df["y"]
    z = df["z"]

    # 3D Plot
    # ax3d.plot(x[::N], y[::N], z[::N], label=planner_labels[planner], color=planner_colors[planner], linewidth=2, linestyle=style["linestyle"])
    if planner != "servo":
        plot_polka_line(ax3d, x, y, z, "#8D5A99", "#e76f51", length=75, label="Cartesian/Pilz Interleaved")
    else:
        ax3d.plot(x, y, z, label=planner_labels[planner], color=planner_colors[planner], linewidth=2.7, linestyle=style["linestyle"])


    ax3d.scatter(df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0], color=planner_colors[planner], s=30, marker="o")
    ax3d.scatter(df["x"].iloc[-1], df["y"].iloc[-1], df["z"].iloc[-1], color=planner_colors[planner], s=30, marker="x")

    # Error metrics
    traj_positions = df[["x", "y", "z"]].values
    traj_orientations = df[["qx", "qy", "qz", "qw"]].values
    tree = KDTree(traj_positions)
    dists, indices = tree.query(ref_positions)
    ref_R = R.from_quat(ref_orientations)
    traj_R = R.from_quat(traj_orientations[indices])
    rel_rot = ref_R.inv() * traj_R
    ang_errors = rel_rot.magnitude() * (180.0 / np.pi)
    
    # Save errors for summary stats
    summary_pos_errors[planner] = dists
    summary_ang_errors[planner] = ang_errors

    # Position error plot
    plt.figure("Position Error per Reference Pose")
    plt.plot(dists, label=planner_labels[planner], color=planner_colors[planner])
    # Angular error plot
    plt.figure("Angular Error per Reference Pose")
    plt.plot(ang_errors, label=planner_labels[planner], color=planner_colors[planner])

# 2D plot formatting
ax2d.set_xlabel("X [m]")
ax2d.set_ylabel("Y [m]")
ax2d.set_title(f"2D Trajectories (X vs Y) for {robot.upper()}")
ax2d.legend()
ax2d.grid(True)
fig2d.tight_layout()

# ax3d.scatter(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], color='black', s=22, marker='x', label="Reference Path")
# ax3d.scatter(
#     ref_positions[:, 0],
#     ref_positions[:, 1],
#     ref_positions[:, 2],
#     facecolors='none',         # Hollow (open) circles
#     edgecolors='black',        # Ring color
#     s=22,                      # Size
#     marker='o',
#     label="Reference Path"
# )

# 3D plot formatting
ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.set_title(f"3D Trajectories for {robot.upper()}", pad=30)
# Remove duplicate legend entries:
handles, labels = ax3d.get_legend_handles_labels()
unique = dict()
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h
ax3d.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=True)

# ax3d.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=True)
fig3d.tight_layout()

# Error plots formatting
plt.figure("Position Error per Reference Pose")
plt.xlabel("Reference Pose Index")
plt.ylabel("Position Error [m]")
plt.title(f"Position Error per Reference Pose ({robot.upper()})")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure("Angular Error per Reference Pose")
plt.xlabel("Reference Pose Index")
plt.ylabel("Angular Error [deg]")
plt.title(f"Angular Error per Reference Pose ({robot.upper()})")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# Optionally print summary stats
for planner in planners:
    print(f"\n{planner_labels[planner]} Planner:")
    print(f"  Mean position error: {np.mean(summary_pos_errors[planner]):.6e} m")
    print(f"  Max position error:  {np.max(summary_pos_errors[planner]):.6e} m")
    print(f"  Mean angular error:  {np.mean(summary_ang_errors[planner]):.3f} deg")
    print(f"  Max angular error:   {np.max(summary_ang_errors[planner]):.3f} deg")
