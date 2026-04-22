import numpy as np

# Paste your matrix here
T_cam_imu = np.array([
  [0.9999792916583793, -0.00630935729775247, 0.0012681738425422806, 0.005094254322024484],
  [0.006299322648145304, 0.9999499966904588, 0.007766765924482546, -0.005454191759019972],
  [-0.0013171137309186491, -0.007758616451432384, 0.9999690340616448, -0.026401055048933034],
  [0.0, 0.0, 0.0, 1.0]
])

# Invert matrix
T_inv = np.linalg.inv(T_cam_imu)

# Print in desired format
flat = T_inv.flatten()

print("data: [", end="")
for i, val in enumerate(flat):
    if i > 0:
        if i % 4 == 0:
            print("\n       ", end="")
        else:
            print(" ", end="")
    print(f"{val:.8f},", end="")
print("]")