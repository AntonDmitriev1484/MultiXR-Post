
from pathlib import Path
import pkgutil
import importlib
import inspect
import os
import json
import csv
import yaml
import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace

import shutil
import math
import copy

from utils.load_rostypes import *
from utils.ros_msg_handlers import *
from utils.math_utils import *
from utils.vicon_utils import *
from utils.optitrack_utils import *


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Example usage:
# python3 post_process.py opti_circle_test1 --opti --slam 20
# keep opti at default frequency, subsample SLAM to 20Hz.

NO_SUBSAMPLE = -10
parser = argparse.ArgumentParser(description="Stream collector")
parser.add_argument("id", type=int)
parser.add_argument("trial_name" , type=str)
parser.add_argument("--calibration_file", "-c", type=str)
parser.add_argument("--opti", nargs="?", const=0, type=float)
parser.add_argument("--slam", nargs="?", const=0, type=float)
# Replace SLAM tracked body in SLAM frame trajectory, with SLAM tracked body in optitrack frame trajectory
parser.add_argument("--align", "-a", action="store_true")
args = parser.parse_args()

ID = args.id

outpath = f'./{ID}/post/{args.trial_name}_post'
out_world = outpath+f'/world/' # Vicon can define apriltags and anchors set up in world frame
os.makedirs(outpath, exist_ok=True)
os.makedirs(out_world, exist_ok=True)

in_slam = f'./{ID}/orbslam/out/{args.trial_name}_nuc{ID}_raw_cam_traj.txt'
in_slam_kf = f'./{ID}/orbslam/out/{args.trial_name}_nuc{ID}_raw_kf_traj.txt'

inpath = f'./{ID}/collect/{args.trial_name}_nuc{ID}_raw'
in_kalibr_dir = inpath+f"/calibration/"
kalibr_files = list(Path(in_kalibr_dir).glob("*.yaml"))
in_kalibr = f"{kalibr_files[0]}"

imu_json = json.load(open(inpath+'/imu_raw.json', 'r'))
uwb_json = json.load(open(inpath+'/uwb_raw.json', 'r'))
metadata = json.load(open(inpath+'/meta.json', 'r'))


# # Filter for messages within bag timestamp range.
START = metadata["start_ns"] * 1e-9
END = metadata["end_ns"] * 1e-9
print(f"Data duration {START} - {END}")


in_opti_bagpath = Path(f"/home/antond2/ros_ws/ros2/{args.trial_name}")
bagpath = Path(f'../collect/ros2/{args.trial_name}')
opti_helmet_traj, opti_anchor_trajectories = load_optitrack(in_opti_bagpath, ID)
opti_data = crop_opti(opti_helmet_traj, START, END)
opti_data = clean_opti(opti_data)




def filtt(arr): # For filtering a json output
    return list(filter(lambda x: (START <= x["t"] <= END), arr)) # Then filter by ros timestamps
def filtt2(arr): # For filtering a CSV output
    return list(filter(lambda x: (START <= x[0] <= END), arr))


### Define all coordinate frames in T
T = define_transforms(in_kalibr) #TODO: Need to revise this method for how optitrack reports

# Position of each UWB anchor transmitter in the optitrack world frame
anchor_positions = compute_anchors(opti_anchor_trajectories, T.T_optiuwb_to_uwbtx)


### Process SLAM data
slam_json = []
body_slam_tum_traj = [] # Preparing for later alignment
body_slam_HTMs = []

if args.slam is not None:
    slam_kf_data = np.loadtxt(in_slam_kf)
    slam_kf_data[:,0] *= 1e-9
    slam_data = np.loadtxt(in_slam)
    slam_data[:,0] *= 1e-9 # Adjust timestamps to be in 's'
    ZERO_TIMESTAMP = slam_data[0][0]

    # TODO: Need to re-do this to give the SLAM body in the SLAM world frame
    def slam_tracked_body_to_my_body(T_cam1_to_sorigin): # SLAM quat gives you the transform from cam1 frame to slam origin
        return T.T_cam1_to_body @ np.linalg.inv(T_cam1_to_sorigin)

    slam_body_poses, slam_body_velocities = aggregate_tracker(slam_tracked_body_to_my_body, slam_data)
    body_slam_HTMs = slam_body_poses
    # Each pose is the IMU pose in the slam frame. i.e. T_slamworld_to_imu. This is what plotting assumes.

    # aggregate_tracker returns [timestamp, flattened HTM]
    # for slam_tum_traj we need to convert back to tum
    for pose in slam_body_poses: body_slam_tum_traj.append(slam_HTM_to_TUM(pose))

    # If we passed a valid frequency to subsample to
    slam_freq = len(slam_data) / (END-START)
    if slam_freq > args.slam > 0:
        skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
        slam_body_poses = np.array(slam_body_poses)
        slam_body_poses = slam_body_poses[::skip] # Finally, subsample to required frequency

        # Convert from nparray to json format
    # Add Vicon poses to all_data
    slam_json = [ {
            "t": float(body_pose[0]),
            "type": "slam_pose",
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": float(body_v[1]),
                    "vy": float(body_v[2]),
                    "vz": float(body_v[3])
            }
        } for body_pose, body_v in zip( list(slam_body_poses), list(slam_body_velocities))]

### Process optitrack data
opti_json = []
body_opti_tum_traj = []
body_opti_HTMs = []
if args.opti is not None:
    def opti_tracked_body_to_my_body(T_head_to_world):
        return T.T_head_to_body @ np.linalg.inv(T_head_to_world)
    
    # def opti_tracked_body_to_my_body(T_head_to_world):
    #     # By default, vicon pose tracking gives you the T_head_to_world
    #     return np.linalg.inv(T_head_to_world)

    opti_body_poses, opti_body_velocities = aggregate_tracker(opti_tracked_body_to_my_body, np.array(opti_data))
    body_opti_HTMs = opti_body_poses
    # Each pose is the IMU pose in the optitrack world frame.

    # aggregate_tracker returns [timestamp, flattened HTM]
    # for opti_tum_traj we need to convert back to tum
    for pose in opti_body_poses: body_opti_tum_traj.append(slam_HTM_to_TUM(pose))

    # If we passed a valid frequency to subsample to
    opti_freq = len(opti_data) / (END-START)
    if opti_freq > args.opti > 0:
        skip = math.ceil(opti_freq / args.opti) # Number of poses to skip in subsampling to synth slam frequency
        opti_body_poses = np.array(opti_body_poses)
        opti_body_poses = opti_body_poses[::skip] # Finally, subsample to required frequency

    opti_json = [ {
            "t": float(body_pose[0]),
            "type": "opti_pose",
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": float(body_v[1]),
                    "vy": float(body_v[2]),
                    "vz": float(body_v[3])
            }
        } for body_pose, body_v in zip( list(opti_body_poses), list(opti_body_velocities))]
    

aligned_slam_json = []
if args.align:
     # Align the body's SLAM trajectory to the Optitrack trajectory
     # Places our SLAM trajectory in a shared coordinate frame
    body_slam_aligned_tum_traj = umeyama_alignment(body_opti_HTMs, body_slam_HTMs)

    def identity(T_body_to_world): # Already in body frame.
        return T_body_to_world
    aligned_slam_body_poses, aligned_slam_body_velocities = aggregate_tracker(identity, np.array(body_slam_aligned_tum_traj))
    # Each pose is the IMU pose in the optitrack world frame.

    # If we passed a valid frequency to subsample to
    slam_freq = len(slam_data) / (END-START)
    if slam_freq > args.slam > 0:
        skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
        aligned_slam_body_poses = np.array(aligned_slam_body_poses)
        aligned_slam_body_poses = aligned_slam_body_poses[::skip] # Finally, subsample to required frequency

    aligned_slam_json = [ {
            "t": float(body_pose[0]),
            "type": "aligned_slam_pose",
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": float(body_v[1]),
                    "vy": float(body_v[2]),
                    "vz": float(body_v[3])
            }
        } for body_pose, body_v in zip( list(aligned_slam_body_poses), list(aligned_slam_body_velocities))]

    slam_json = aligned_slam_json



# If we're using real UWB ranges, but have no compass
# We interpolate on SLAM poses to match a synthetic orientation to that UWB range
# assisted_uwb_json = []
# if args.map_vicon_to_uwb:
#     assisted_uwb_json = aggregate_assisted_uwb(uwb_json, vicon_tracked_body_to_my_body, np.array(vicon_data[vicon_name]), 100)

# Compose the final factor graph dataset
all_data = uwb_json + imu_json + opti_json + slam_json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return super().default(obj)

### Copy all world information: T, anchors, apriltags, to output

out_anchors = open(f'{outpath}/anchors.json', 'w')
json.dump(anchor_positions, out_anchors, cls=NumpyEncoder, indent=1)


with open(f'{outpath}/transforms.json', 'w') as fs: json.dump(vars(T), fs, cls=NumpyEncoder, indent=1)

# Run sanity check to make sure measurements are at the frequency we expect them to be before testing in the graph

print("Checking frequency of real data")
print(f" Measured UWB frequency {len(uwb_json) / (END-START)}")
print(f" Measured optitrack frequency {len(opti_data) / (END-START)}")
print(f" Measured SLAM frequency {len(slam_data) / (END-START)}")

print("Checking frequency of subsampled output")
print(f" Measured subsampled optitrack frequency {len(opti_json) / (END-START)}")
print(f" Measured subsampled SLAM frequency {len(slam_json) / (END-START)}")


# Filter to make sure all messages ( and data jsons ) fall within the ROS recording time interval, (because some of them don't apparently)
all_data = filtt(all_data)
all_data = sorted(all_data, key=lambda x: x["t"])

json.dump(all_data, open(outpath+"/all.json", 'w'), cls=NumpyEncoder, indent=1)
# json.dump(priors, open(outpath+"/priors.json", 'w'), cls=NumpyEncoder, indent=1)
