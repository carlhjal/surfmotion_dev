import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ament_index_python.packages import get_package_share_directory

# --- USER SETTINGS ---
robot = "fanuc"    # e.g. "ur20"
planner = "pilz"  # e.g. "cart", "pilz", "servo"
data_dir = os.path.join(get_package_share_directory("eval_tools"), "output", f"{planner}_{robot}")

eef_file = sorted(glob.glob(os.path.join(data_dir, "eef*")))[0]
df = pd.read_csv(eef_file)
t = df["time"].values
pos = df[["x", "y", "z"]].values

# --- Euler (Finite Difference) ---
dt = np.diff(t)
vel_euler = np.diff(pos, axis=0) / dt[:, None]
acc_euler = np.diff(vel_euler, axis=0) / dt[1:, None]
jerk_euler = np.diff(acc_euler, axis=0) / dt[2:, None]

# --- Savitzky-Golay ---
window = 31 if len(t) > 31 else (len(t) // 2) * 2 + 1  # odd, <= len
poly = 3
vel_savgol = np.empty_like(pos)
acc_savgol = np.empty_like(pos)
jerk_savgol = np.empty_like(pos)
for axis in range(3):
    vel_savgol[:, axis]  = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=1, delta=t[1]-t[0], mode="interp")
    acc_savgol[:, axis]  = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=2, delta=t[1]-t[0], mode="interp")
    jerk_savgol[:, axis] = savgol_filter(pos[:, axis], window_length=window, polyorder=poly, deriv=3, delta=t[1]-t[0], mode="interp")

# --- Magnitude (Euclidean norm) ---
def norm(a):
    return np.linalg.norm(a, axis=1)

# --- PLOTTING ---

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(t[1:], norm(vel_euler), 'b--', label='Euler')
axes[0].plot(t, norm(vel_savgol), 'r', label='Savitzky-Golay')
axes[0].set_ylabel("Velocity Magnitude [m/s]")
axes[0].set_title(f"Velocity Magnitude ({robot.upper()}, {planner})")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t[2:], norm(acc_euler), 'b--', label='Euler')
axes[1].plot(t, norm(acc_savgol), 'r', label='Savitzky-Golay')
axes[1].set_ylabel("Acceleration Magnitude [m/s²]")
axes[1].set_title("Acceleration Magnitude")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t[3:], norm(jerk_euler), 'b--', label='Euler')
axes[2].plot(t, norm(jerk_savgol), 'r', label='Savitzky-Golay')
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Jerk Magnitude [m/s³]")
axes[2].set_title("Jerk Magnitude")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
