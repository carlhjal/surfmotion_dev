import numpy as np
import pandas as pd

planners = ['Cartesian', 'Pilz', 'Servo']

# 1. Table data (your values)
err_rms_pos = np.array([5.47e-5, 5.71e-5, 0.0370])      # Lower is better
err_rms_ang = np.array([0.02, 0.015, 41.515])           # Lower is better
smoothness = np.array([3.23, 13.96, 26.14])             # Lower is better
vel_consistency = np.array([0.506, 0.749, 0.73])        # Higher is better

# 2. Normalize each metric (0=worst, 1=best)
def norm_low_is_good(x):
    return 1 - (x - np.min(x)) / (np.max(x) - np.min(x))
def norm_high_is_good(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

norm_pos = norm_low_is_good(err_rms_pos)
norm_ang = norm_low_is_good(err_rms_ang)
norm_smooth = norm_low_is_good(smoothness)
norm_velc = norm_high_is_good(vel_consistency)

# 3. Aggregate score (average, but you can use weighted if you want)
aggregate = (norm_pos + 1*norm_ang + 1*norm_smooth + 1*norm_velc) / 4

# 4. Results as a table
df = pd.DataFrame({
    'Planner': planners,
    'RMS Pos. Err. [m]': err_rms_pos,
    'RMS Ang. Err. [deg]': err_rms_ang,
    'Smoothness [1/s^2]': smoothness,
    'Vel Consistency': vel_consistency,
    'Score (0=bad, 1=good)': aggregate
})

# Optional: sort by best
df = df.sort_values('Score (0=bad, 1=good)', ascending=False).reset_index(drop=True)

print(df.to_string(index=False, float_format="{:.4f}".format))
