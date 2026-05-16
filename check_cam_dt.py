import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DIR = Path("/home/antond2/Desktop/Research/MultiXR-Post/3/collect/opti_multi1_free_circle_nuc3_raw/infra1_raw")

# Get all PNG files
files = sorted(DIR.glob("*.png"))

# Extract timestamps from filenames
# Assumes filenames are like:
# 1778865507.123456.png
timestamps = np.array(
    [float(f.stem) for f in files],
    dtype=float
)

# Compute delta times
dts = np.diff(timestamps)

# Sample indices for x-axis
samples = np.arange(len(dts))

# Scatter plot
plt.figure(figsize=(12, 5))
plt.scatter(samples, dts, s=5)

plt.title("Camera Frame Delta Times")
plt.xlabel("Frame Number")
plt.ylabel("Delta Time (s)")

plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
print(f"Frames: {len(timestamps)}")
print(f"Mean dt: {np.mean(dts):.6f} s")
print(f"Std  dt: {np.std(dts):.6f} s")
print(f"Min  dt: {np.min(dts):.6f} s")
print(f"Max  dt: {np.max(dts):.6f} s")
print(f"Estimated frequency: {1.0 / np.mean(dts):.2f} Hz")

# Optional: detect suspicious gaps
threshold = 1.5 * np.mean(dts)

bad = np.where(dts > threshold)[0]

print(f"\nLarge gaps (> {threshold:.6f} s): {len(bad)}")

for idx in bad[:20]:
    print(
        f"Gap at frame {idx}: "
        f"{timestamps[idx]:.6f} -> {timestamps[idx+1]:.6f} "
        f"(dt={dts[idx]:.6f})"
    )