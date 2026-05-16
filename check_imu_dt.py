import json
import numpy as np
import matplotlib.pyplot as plt

FILE = "/home/antond2/Desktop/Research/MultiXR-Post/3/collect/opti_multi1_free_circle_nuc3_raw/imu_raw.json"

# Load JSON
with open(FILE, "r") as f:
    data = json.load(f)

# Extract timestamps
timestamps = np.array([entry["t"] for entry in data], dtype=float)

# Compute delta times
dts = np.diff(timestamps)

# Sample indices for x-axis
samples = np.arange(len(dts))

# Scatter plot
plt.figure(figsize=(12, 5))
plt.scatter(samples, dts, s=5)

plt.title("IMU Delta Times")
plt.xlabel("Sample Number")
plt.ylabel("Delta Time (s)")

plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
print(f"Samples: {len(timestamps)}")
print(f"Mean dt: {np.mean(dts):.6f} s")
print(f"Std  dt: {np.std(dts):.6f} s")
print(f"Min  dt: {np.min(dts):.6f} s")
print(f"Max  dt: {np.max(dts):.6f} s")
print(f"Estimated frequency: {1.0 / np.mean(dts):.2f} Hz")