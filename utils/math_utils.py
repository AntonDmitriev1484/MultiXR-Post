

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace
import yaml

# Imports do work, even if not detected
from evo.core import trajectory
from evo.core import sync
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
import matplotlib.pyplot as plt

# Note; By default SLAM tracked body frame is the left camera
# A SLAM trajectory is T_leftcam_in_slamframe
def define_transforms(in_kalibr):
    T = SimpleNamespace()

    # Transform from optitrack UWB anchor markers to UWB antenna.
    # TODO: Verify by hand
    T.T_optiuwb_to_uwbtx = np.eye(4) # Probably better to express as a vector in the vUWB frame
    T.T_optiuwb_to_uwbtx[:3, 3] = [0.045, 0.04, 0] # 3cm down along x-axis.

    # 'Head' refers to the vicon tracked head pose
    T.T_imu_to_body = np.eye(4)
    T.T_body_to_imu = np.linalg.inv(T.T_imu_to_body)

    # Transform from head tracking optitrack markers to left camera
    # TODO: Verify by hand
    T.T_head_to_cam1 = np.array(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0.0175],
         [0, 0, -1, -0.08],
         [0, 0, 0, 1]]
    )

    with open(in_kalibr, 'r') as fs: calibration = yaml.safe_load(fs)
    T.T_imu_to_cam1 = np.array(calibration['cam0']['T_cam_imu'])
    T.T_cam1_to_body = T.T_imu_to_body @ np.linalg.inv(T.T_imu_to_cam1)
    T.T_head_to_body = T.T_cam1_to_body @ T.T_head_to_cam1

    # Would only change if you're using vicon2gt
    T.T_inertial_to_world = np.eye(4)

    # Transform from head tracking optitrack markers to decawave antenna
    # TODO: Verify by hand

    T.T_head_to_decawave = np.array(
        [
         [1, 0, 0, 0.01],
         [0, 1, 0, 0.0525],
         [0, 0, 1, 0.0175],
         [0, 0, 0, 1]
        ]
    )

    # From old Vicon code
    # T_decawave_to_head = np.eye(4)
    # T_decawave_to_head[:3,3] = np.array([-0.01, -0.0175, 0.0525])
    # T.T_head_to_decawave = np.linalg.inv(T_decawave_to_head)
    
    # GTSAM expects the "pose of object in body frame"
    # i.e. T_decawave_to_body, but I can do that invert in C++
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

def dump_stats(traj_ref_sync, traj_est_sync):

    # Translation APE
    ape_metric_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric_trans.process_data((traj_ref_sync, traj_est_sync))
    ape_stats = ape_metric_trans.get_all_statistics()
    print(f"    Translation APE,\n\t{ape_stats["mean"]=},\n\t{ape_stats["rmse"]=}")

    # Rotation APE
    ape_metric_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_metric_rot.process_data((traj_ref_sync, traj_est_sync))
    ape_stats = ape_metric_rot.get_all_statistics()
    # print(f" Rotational APE {json.dumps(ape_stats, indent=1)}")
    print(f"    Rotation APE,\n\t{ape_stats["mean"]=},\n\t{ape_stats["rmse"]=}")

# These get passed in as lists of 1 x 17 numpy arrays
def umeyama_alignment(body_opti_HTMs, body_slam_HTMs):

    # Per examples here: https://github.com/MichaelGrupp/evo/blob/master/examples/alignment_demo.py

    body_opti_HTMs = np.array(body_opti_HTMs)
    body_slam_HTMs = np.array(body_slam_HTMs)

    opti_ts = body_opti_HTMs[:, 0]
    slam_ts = body_slam_HTMs[:, 0]

    opti_poses = np.linalg.inv(body_opti_HTMs[:, 1:17].reshape((-1, 4,4)))
    slam_poses = np.linalg.inv(body_slam_HTMs[:, 1:17].reshape((-1, 4,4)))

    # Convert positions to evo format
    opti_positions = opti_poses[:, :3, 3]
    slam_positions = slam_poses[:, :3, 3]

    # Convert rotations to evo format
    opti_rot_mats = opti_poses[:, :3, :3]
    slam_rot_mats = slam_poses[:, :3, :3]
    opti_quats_xyzw = R.from_matrix(opti_rot_mats).as_quat()
    slam_quats_xyzw = R.from_matrix(slam_rot_mats).as_quat()
    def xyzw_to_wxyz(q):
        return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
    opti_quats = xyzw_to_wxyz(opti_quats_xyzw)
    slam_quats = xyzw_to_wxyz(slam_quats_xyzw)

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


    # Time synchronization
    traj_ref, traj_est = sync.associate_trajectories(opti_traj, slam_traj, max_diff = 0.01)

    # Align (SE3)
    traj_est.align(traj_ref, correct_scale=True)

    print(f"Umeyama Alignment Results:")
    dump_stats(traj_ref, traj_est)

    # traj_est is now SLAM trajectory aligned to optitrack world frame

    traj_est_out = []

    rots = R.from_quat(traj_est.orientations_quat_wxyz[:, [1,2,3,0]])  # wxyz → xyzw

    for t, p, r in zip(
        traj_est.timestamps,
        traj_est.positions_xyz,
        rots
    ):
        # inverse rotation
        r_inv = r.inv()

        # inverse translation: -R^T t
        p_inv = -r_inv.apply(p)

        # quaternion back to xyzw
        qx, qy, qz, qw = r_inv.as_quat()

        # append
        traj_est_out.append([
            t,
            p_inv[0], p_inv[1], p_inv[2],
            qx, qy, qz, qw
        ])

    return traj_est_out