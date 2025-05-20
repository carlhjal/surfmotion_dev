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

# Load data
dir = os.path.join(get_package_share_directory("eval_tools"), "output", subdirs["ur20"]["cart"])
eef_files = glob.glob(os.path.join(dir, "eef*"))
joint_files = glob.glob(os.path.join(dir, "joint*"))
poses = json_load(os.path.join(dir, "poses.json"))
ref_positions = np.array([[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in poses])
ref_orientations = np.array([[p["orientation"]["x"], p["orientation"]["y"], p["orientation"]["z"], p["orientation"]["w"]] for p in poses])

dfs = [pd.read_csv(f) for f in eef_files]
fig, ax = plt.subplots()

for i, path in enumerate(eef_files):
    df = pd.read_csv(path)
    df["time"] = df["time"] - df["time"].iloc[0]  # Normalize time

    label = f"Run {i+1}"
    ax.plot(df["x"], df["y"], label=label, linestyle='--', marker='o', markersize=2)
    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color="green", s=30)
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", s=30)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("2D Trajectories (X vs Y) from Multiple Runs")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()


ref = dfs[0]
plt.figure()
for i, df in enumerate(dfs[1:], start=2):
    df["time"] = df["time"] - df["time"].iloc[0]  # Normalize time
    min_len = min(len(df), len(ref))

    displacement = np.linalg.norm(
        df[["x", "y", "z"]].values[:min_len] - ref[["x", "y", "z"]].values[:min_len],
        axis=1
    )
    time = df["time"].iloc[:min_len]  # Truncate time to match displacement
    plt.plot(time, displacement, label=f"Run {i} vs Run 1")
    
    smoothed = savgol_filter(displacement, window_length=400, polyorder=5)
    plt.plot(df["time"].iloc[:len(smoothed)], smoothed, label=f"Filtered Run {i} vs Run 1")
    
    rms = np.sqrt(np.mean(displacement**2))
    print(f"RMS error: {rms:.6e} m")

plt.xlabel("Time [s]")
plt.ylabel("Euclidean Distance [m]")
plt.title("Deviation from Reference Trajectory (Run 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


pos_errors_all = []
ang_errors_all = []

print("\n--- Deviation from Reference Poses ---")
for i, df in enumerate(dfs):
    traj_positions = df[["x", "y", "z"]].values
    traj_orientations = df[["qx", "qy", "qz", "qw"]].values

    tree = KDTree(traj_positions)
    dists, indices = tree.query(ref_positions)

    # Orientation error
    ref_R = R.from_quat(ref_orientations)
    traj_R = R.from_quat(traj_orientations[indices])
    rel_rot = ref_R.inv() * traj_R
    ang_errors = rel_rot.magnitude() * (180.0 / np.pi)
    
    pos_errors_all.append(dists)
    ang_errors_all.append(ang_errors)

    print(f"\nRun {i+1}:")
    print(f"  Mean position error: {np.mean(dists):.6e} m")
    print(f"  Max position error:  {np.max(dists):.6e} m")
    print(f"  Mean angular error:  {np.mean(ang_errors):.3f} deg")
    print(f"  Max angular error:   {np.max(ang_errors):.3f} deg")
    
# --- Plot position errors ---
plt.figure()
for i, dists in enumerate(pos_errors_all):
    plt.plot(dists, label=f"Run {i+1}")
plt.xlabel("Reference Pose Index")
plt.ylabel("Position Error [m]")
plt.title("Position Error per Reference Pose")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Plot angular errors ---
plt.figure()
for i, ang in enumerate(ang_errors_all):
    plt.plot(ang, label=f"Run {i+1}")
plt.xlabel("Reference Pose Index")
plt.ylabel("Angular Error [deg]")
plt.title("Angular Error per Reference Pose")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i, path in enumerate(eef_files):
#     df = pd.read_csv(path)
#     df["time"] = df["time"] - df["time"].iloc[0]  # Normalize time

#     label = f"Run {i+1}"
#     ax.plot(df["x"], df["y"], df["z"], label=label)
#     ax.scatter(df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0], s=30, color="green")
#     ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], df["z"].iloc[-1], s=30, color="red")

# # Labels
# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_zlabel("Z [m]")
# ax.set_title("3D Trajectories from Multiple Runs")
# ax.legend()
# plt.tight_layout()
# plt.show()