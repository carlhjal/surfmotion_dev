import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ament_index_python.packages import get_package_share_directory

robot = "ur20"
planners = ["cart", "pilz", "servo"]
planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
planner_colors = {
    "cart": "#8D5A99",  # Purple
    "pilz": "#e76f51",  # Orange
    "servo": "#2186a5"  # Teal
}

def norm(a): return np.linalg.norm(a, axis=1)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
# axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[2].set_yscale("log")

for planner in planners:
    data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", f"{planner}_{robot}")
    eef_file = sorted(glob.glob(os.path.join(data_dir, "eef*")))[0]
    df = pd.read_csv(eef_file)
    t = df["time"].values
    t = t - t[0]  # normalize so each starts at 0
    pos = df[["x", "y", "z"]].values
    color = planner_colors[planner]
    label = planner_labels[planner]

    # Euler method
    dt = np.diff(t)
    vel_euler = np.diff(pos, axis=0) / dt[:, None]
    acc_euler = np.diff(vel_euler, axis=0) / dt[1:, None]
    jerk_euler = np.diff(acc_euler, axis=0) / dt[2:, None]

    # Savitzky-Golay
    window = 31 if len(t) > 31 else (len(t) // 2) * 2 + 1  # odd, <= len
    poly = 3
    vel_savgol = np.empty_like(pos)
    acc_savgol = np.empty_like(pos)
    jerk_savgol = np.empty_like(pos)
    for axis in range(3):
        vel_savgol[:, axis]  = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=1, delta=t[1]-t[0], mode="interp")
        acc_savgol[:, axis]  = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=2, delta=t[1]-t[0], mode="interp")
        jerk_savgol[:, axis] = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=3, delta=t[1]-t[0], mode="interp")

    # Velocity
    axes[0].plot(t[1:], norm(vel_euler), linestyle="--", color=color, alpha=0.65, label=f"{label} (Euler)")
    # axes[0].plot(t, norm(vel_savgol), linestyle="-", color=color, label=f"{label} (Savitzky)")
    # Acceleration
    axes[1].plot(t[2:], norm(acc_euler), linestyle="--", color=color, alpha=0.65, label=f"{label} (Euler)")
    axes[1].plot(t, norm(acc_savgol), linestyle="-", color=color, label=f"{label} (Savitzky)")
    # Jerk
    axes[2].plot(t[3:], norm(jerk_euler), linestyle="--", color=color, alpha=0.65, label=f"{label} (Euler)")
    axes[2].plot(t, norm(jerk_savgol), linestyle="-", color=color, label=f"{label} (Savitzky)")

axes[0].set_ylabel("Velocity [m/s]")
axes[0].set_title(f"Velocity Magnitude ({robot.upper()}): Euler vs. Savitzky-Golay")
axes[0].legend(ncol=2)
axes[0].grid(True)

axes[1].set_ylabel("Acceleration [m/s²]")
axes[1].set_title("Acceleration Magnitude")
axes[1].legend(ncol=2)
axes[1].grid(True)

axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Jerk [m/s³]")
axes[2].set_title("Jerk Magnitude")
axes[2].legend(ncol=2)
axes[2].grid(True)

plt.tight_layout()
plt.show()
