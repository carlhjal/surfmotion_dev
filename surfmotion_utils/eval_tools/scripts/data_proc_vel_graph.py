import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from scipy.signal import savgol_filter

robot = "kuka"
planners = ["cart", "pilz", "servo"]
planner_labels = {"cart": "Cartesian", "pilz": "Pilz", "servo": "Servo"}
planner_colors = {"cart": "#8D5A99", "pilz": "#e76f51", "servo": "#2186a5"}
# subdirs = {
#     "ur20": {"cart": "cart_ur20", "pilz": "pilz_ur20", "servo": "servo_ur20"}
# }
subdirs = {
    "ur20": {"cart": "cart_ur20", "pilz": "pilz_ur20", "servo": "servo_ur20"},
    "fanuc": {"cart": "cart_fanuc", "pilz": "pilz_fanuc", "servo": "servo_fanuc"},
    "ur5": {"cart": "cart_ur5", "pilz": "pilz_ur5", "servo": "servo_ur5"},
    "kuka": {"cart": "cart_kuka", "pilz": "pilz_kuka", "servo": "servo_kuka"}
}
def load_xyz_time(path):
    df = pd.read_csv(path)
    pos = df[["x", "y", "z"]].values
    time = df["time"].values
    return pos, time

def euler_diff(signal, time):
    dt = np.diff(time)
    out = np.diff(signal, axis=0) / dt[:, None]
    # Pad to keep dimensions
    out = np.vstack([out[0], out])
    return out

def magnitude(arr):
    return np.linalg.norm(arr, axis=1)

fig_width = 8
fig_height = 4.5

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})

fig, axes = plt.subplots(1, 1, figsize=(fig_width, fig_height), sharex=True)
# fig, axes = plt.subplots()
print("RMS Jerk Magnitude Table")
print(f"{'Planner':<12} {'Euler':>12} {'Savitzky':>12} {'Mixed':>12}")

for planner in planners:
    data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs[robot][planner])
    eef_files = sorted(glob.glob(os.path.join(data_dir, "eef*")))
    pos, time = load_xyz_time(eef_files[0])
    time = time - time[0]  # normalize so each starts at 0
    dt = np.mean(np.diff(time))
    
    # --- Euler ---
    vel_euler = euler_diff(pos, time)
    acc_euler = euler_diff(vel_euler, time)
    jerk_euler = euler_diff(acc_euler, time)

    vel = euler_diff(pos, time)
    # vel = savgol_filter(vel, window_length=31, polyorder=3, axis=0)  # or any filter

    acc = euler_diff(vel, time)
    # acc = savgol_filter(acc, window_length=31, polyorder=3, axis=0)

    jerk = euler_diff(acc, time)
    # jerk = savgol_filter(jerk, window_length=31, polyorder=3, axis=0)
    
    vel_euler = vel
    acc_euler = acc
    jerk_euler = jerk


    # --- Savitzky-Golay ---
    window = 31 if len(time) > 25 else 5
    poly = 3
    vel_savgol = savgol_filter(pos, window, poly, deriv=1, delta=dt, axis=0)
    acc_savgol = savgol_filter(pos, window, poly, deriv=2, delta=dt, axis=0)
    jerk_savgol = savgol_filter(pos, window, poly, deriv=3, delta=dt, axis=0)

    # --- Mixed ---
    acc_mixed = savgol_filter(vel_euler, window, poly, deriv=1, delta=dt, axis=0)
    jerk_mixed = savgol_filter(vel_euler, window, poly, deriv=2, delta=dt, axis=0)

    # --- Magnitudes ---
    v_euler_mag = magnitude(vel_euler)
    v_savgol_mag = magnitude(vel_savgol)
    a_euler_mag = magnitude(acc_euler)
    a_savgol_mag = magnitude(acc_savgol)
    j_euler_mag = magnitude(jerk_euler)
    j_savgol_mag = magnitude(jerk_savgol)
    a_mixed_mag = magnitude(acc_mixed)
    j_mixed_mag = magnitude(jerk_mixed)

    # Compute RMS jerk for each method
    rms_jerk_euler = np.sqrt(np.mean(j_euler_mag**2))
    rms_jerk_savgol = np.sqrt(np.mean(j_savgol_mag**2))
    rms_jerk_mixed = np.sqrt(np.mean(j_mixed_mag**2))
    print(f"{planner_labels[planner]:<12} {rms_jerk_euler:12.4e} {rms_jerk_savgol:12.4e} {rms_jerk_mixed:12.4e}")

    color = planner_colors[planner]
    label_base = planner_labels[planner]
    
    # axes[0].plot(time, v_euler_mag, linestyle="--", color=color, alpha=0.5, label=f"{label_base} (Euler)")
    axes.plot(time, v_savgol_mag, linestyle="-", color=color, alpha=0.85, label=f"{label_base}")

    # axes[1].plot(time, a_euler_mag, linestyle="--", color=color, alpha=0.5, label=f"{label_base} (Euler)")
    # axes[1].plot(time, a_savgol_mag, linestyle="-", color=color, alpha=0.85, label=f"{label_base} (Savitzky)")
    # axes[1].plot(time, a_mixed_mag, linestyle=":", color=color, alpha=0.85, label=f"{label_base} (Mixed)")

    # axes[2].plot(time, j_euler_mag, linestyle="--", color=color, alpha=0.5, label=f"{label_base} (Euler)")
    # axes[2].plot(time, j_savgol_mag, linestyle="-", color=color, alpha=0.85, label=f"{label_base} (Savitzky)")
    # axes[2].plot(time, j_mixed_mag, linestyle=":", color=color, alpha=0.85, label=f"{label_base} (Mixed)")

axes.set_ylabel("Velocity [m/s]")
# axes.set_title(f"Velocity Magnitude")
# axes[1].set_ylabel("Acceleration [m/s²]")
# axes[1].set_title("Acceleration Magnitude")
# axes[2].set_ylabel("Jerk [m/s³]")
# axes[2].set_title("Jerk Magnitude")
axes.set_xlabel("Time [s]")

axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=True)
fig.suptitle("Time-Velocity (Kuka)", fontsize=18, y=0.9, x=0.53)
# for i, ax in enumerate(axes):
#     ax.grid(True)
#     if i > 0:
#         ax.set_yscale("log")

plt.tight_layout()
plt.show()

