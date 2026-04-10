

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace
import yaml

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

    T_cam1_to_head = np.array([[-1 , 0, 0, 0.0175],
                            [0, 0, -1, -0.08],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]])
    T.T_head_to_cam1 = np.linalg.inv(T_cam1_to_head)

    with open(in_kalibr, 'r') as fs: calibration = yaml.safe_load(fs)
    T.T_imu_to_cam1 = np.array(calibration['cam0']['T_cam_imu'])
    T.T_cam1_to_body = T.T_imu_to_body @ np.linalg.inv(T.T_imu_to_cam1)
    T.T_head_to_body = T.T_cam1_to_body @ T.T_head_to_cam1 # Seems to work better?


    T.T_head_to_body = np.eye(4)
    T.T_inertial_to_world = np.eye(4)


    T_decawave_to_head = np.eye(4)
    T_decawave_to_head[:3,3] = np.array([-0.01, -0.0175, 0.0525])
    T.T_head_to_decawave = np.linalg.inv(T_decawave_to_head)
    T.T_body_to_decawave = T.T_head_to_decawave @ np.linalg.inv(T.T_head_to_body)



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