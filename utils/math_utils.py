

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace
import yaml

# Imports do work, even if not detected
from evo.core import trajectory
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
import matplotlib.pyplot as plt


def define_transforms(in_kalibr):
    T = SimpleNamespace()

    # Transform from vicon marker on helmet, to center of RS camera (body frame)

    # Vicon coordinate frames are marked with a 'v'

    #Transform from vicon marker on anchor, to the center of the DW1000 UWB chip
    T.T_vuwb_to_uwbtx = np.eye(4) # Probably better to express as a vector in the vUWB frame
    T.T_vuwb_to_uwbtx[:3, 3] = [0.035, 0, 0] # 3cm down along x-axis.


    # The SLAM tracked body is the left camera.

    # 'Head' refers to the vicon tracked head pose
    T.T_imu_to_body = np.eye(4)
    T.T_body_to_imu = np.linalg.inv(T.T_imu_to_body)

    # T_cam1_to_head = np.array([[-1 , 0, 0, 0.0175],
    #                         [0, 0, -1, -0.08],
    #                         [0, -1, 0, 0],
    #                         [0, 0, 0, 1]])
    # T.T_head_to_cam1 = np.linalg.inv(T_cam1_to_head)

    #TODO: Verify by hand
    T.T_head_to_cam1 = np.array(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0.0175],
         [0, 0, -1, -0.08],
         [0, 0, 0, 1]]
    )

    #"Down by 2 deg"
    # T.extra_rot = np.array([
    #      [0.9993908270190958, -0.03489949670250097, 0, 0],
    #      [0.03489949670250097, 0.9993908270190958, 0, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]
    # ])

    #"Up by 2 deg"
    # T.extra_rot = np.array([
    #      [0.9993908270190958, 0.03489949670250097, 0, 0],
    #      [-0.03489949670250097, 0.9993908270190958, 0, 0],
    #      [0, 0, 1, 0],
    #      [0, 0, 0, 1]
    # ])

    with open(in_kalibr, 'r') as fs: calibration = yaml.safe_load(fs)
    T.T_imu_to_cam1 = np.array(calibration['cam0']['T_cam_imu'])
    T.T_cam1_to_body = T.T_imu_to_body @ np.linalg.inv(T.T_imu_to_cam1)
    # T.T_head_to_body = T.T_cam1_to_body @ T.extra_rot @ T.T_head_to_cam1
    T.T_head_to_body = T.T_cam1_to_body @ T.T_head_to_cam1

    # T.T_head_to_body = np.linalg.inv(T.T_imu_to_cam1) @ T.T_head_to_cam1 @ n


    T.T_inertial_to_world = np.eye(4)


    T_decawave_to_head = np.eye(4)
    T_decawave_to_head[:3,3] = np.array([-0.01, -0.0175, 0.0525])
    T.T_head_to_decawave = np.linalg.inv(T_decawave_to_head)
    T.T_body_to_decawave = T.T_head_to_decawave @ np.linalg.inv(T.T_head_to_body)

    return T


def slam_quat_to_HTM(nparr): # Doesnt timestamp
    translation = nparr[1:4]
    quat = nparr[4:8]
    
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()

    # Assemble homogeneous transformation matrix (4x4)
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = translation

    return H

def HTM_to_TUM(T): # 2D pose matrix to TUM format
    # Extract translation and rotation
    t = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

    # Non timestamped
    return [ t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]]

def slam_HTM_to_TUM(nparr): # Same as HTM to TUM but it handles timestamped
    if len(nparr) != 17:
        print(nparr)
        raise ValueError("Expected 17 elements: [timestamp, 16 HTM elements]")

    timestamp = nparr[0]
    T_flat = nparr[1:]
    T = np.array(T_flat).reshape((4, 4))

    # Extract translation and rotation
    t = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

    return [timestamp, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]]

# Expects data to be input as an HTM
# Pose can be passed in any frame, but be mindful of the SLERP left hand coordinate system problem
# Pass a target_timestamp, first < target < second
def interpolate_pose(first_pose, first_timestamp, second_pose, second_timestamp, target_timestamp, n_points):

    # Now interpolate between these two poses
    interp_interval = [first_timestamp, second_timestamp]
    interp_timestamps = np.linspace(first_timestamp, second_timestamp, n_points)

    # Use Slerp to interpolate on SO(3) rotations
    interp_rots = R.from_matrix([first_pose[:3, :3], second_pose[:3, :3]])
    slurpy = Slerp(interp_interval, interp_rots)
    interpolated_rotations = slurpy(interp_timestamps)

    # Use linspace to interpolate on R3 positions
    interpolated_positions = np.linspace(first_pose[:3, 3], second_pose[:3, 3], n_points)

    # Fetch the closest interpolation timestamp to the uwb measurement, and map that interpolated pose to the measurement
    idx_match = np.argmin(np.abs(interp_timestamps - target_timestamp))

    interp_pose = np.eye(4)
    interp_pose[:3,:3] = interpolated_rotations[idx_match].as_matrix()
    interp_pose[:3, 3] = interpolated_positions[idx_match]

    return interp_pose


# These get passed in as lists of 1 x 17 numpy arrays
def umeyama_alignment(body_opti_HTMs, body_slam_HTMs):

    # Per examples here: https://github.com/MichaelGrupp/evo/blob/master/examples/alignment_demo.py

    body_opti_HTMs = np.array(body_opti_HTMs)
    body_slam_HTMs = np.array(body_slam_HTMs)

    opti_ts = body_opti_HTMs[:, 0]
    slam_ts = body_slam_HTMs[:, 0]

    opti_poses = body_opti_HTMs[:, 1:17].reshape((-1, 4,4))
    slam_poses = body_slam_HTMs[:, 1:17].reshape((-1, 4,4))

    # Convert positions to evo format
    opti_positions = opti_poses[:, :3, 3]
    slam_positions = slam_poses[:, :3, 3]

    # Convert rotations to evo format
    opti_rot_mats = opti_poses[:, :3, :3]
    slam_rot_mats = slam_poses[:, :3, :3]
    # opti_quats_xyzw = R.from_matrix(opti_rot_mats).as_quat()
    # slam_quats_xyzw = R.from_matrix(slam_rot_mats).as_quat()
    opti_quats_xyzw = R.from_matrix(opti_rot_mats).inv().as_quat()
    slam_quats_xyzw = R.from_matrix(slam_rot_mats).inv().as_quat()
    def xyzw_to_wxyz(q):
        return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
    opti_quats = xyzw_to_wxyz(opti_quats_xyzw)
    slam_quats = xyzw_to_wxyz(slam_quats_xyzw)

    print(f"{slam_poses[100, :, :]=}")
    print(f"{slam_rot_mats[100, :, :]=}")
    print(f"{slam_positions[100, :]=}")
    print(f"{slam_ts[100]=}")

    opti_traj = PoseTrajectory3D(
        positions_xyz=opti_positions,
        orientations_quat_wxyz=opti_quats,
        timestamps=opti_ts
    )

    slam_traj = PoseTrajectory3D(
        positions_xyz=slam_positions,
        orientations_quat_wxyz=slam_quats,
        timestamps=slam_ts
    )

    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode, subplot_arg=221)
    # plot.traj(ax, plot_mode, traj_ref, "--", "gray")
    plot.traj(ax, plot_mode, slam_traj, "-", "blue")
    fig.axes.append(ax)
    plt.title("SLAM traj object before alignment")


    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode, subplot_arg=221)
    # plot.traj(ax, plot_mode, traj_ref, "--", "gray")
    plot.traj(ax, plot_mode, opti_traj, "-", "green")
    fig.axes.append(ax)
    plt.title("Optitrack traj object before alignment")


    # Time synchronization
    traj_ref, traj_est = sync.associate_trajectories(opti_traj, slam_traj, max_diff = 0.01)

    # Align (SE3)
    traj_est.align(traj_ref, correct_scale=False)


    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode, subplot_arg=221)
    # plot.traj(ax, plot_mode, traj_ref, "--", "gray")
    plot.traj(ax, plot_mode, traj_est, "-", "blue")
    fig.axes.append(ax)
    plt.title("SLAM traj object after alignment")


    plt.show()

    # traj_est is now SLAM trajectory aligned to optitrack world frame

    traj_est_out = []
    # Convert back to list of TUM poses
    for t, p, q in zip(
        traj_est.timestamps,
        traj_est.positions_xyz,
        traj_est.orientations_quat_wxyz
    ):
        # convert wxyz → xyzw
        qx, qy, qz, qw = q[1], q[2], q[3], q[0]
        traj_est_out.append([t, p[0], p[1], p[2], qx, qy, qz, qw])

    return traj_est_out