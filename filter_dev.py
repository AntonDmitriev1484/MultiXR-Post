
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
from utils.uwb_utils import *


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return super().default(obj)


def post_process(args):

    ID = args.id
    outpath = f'./{ID}/post/{args.trial_name}_post'
    out_world = outpath+f'/world/'
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(out_world, exist_ok=True)

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


    # if args.opti:
    in_opti_bagpath = Path(f"/home/antond2/ros_ws/ros2/{args.trial_name}")
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

    ### Process optitrack data
    opti_json = []
    body_opti_tum_traj = [] # Explain: My default pose output so I can automate evo in eval.py
    body_opti_inv_tum_traj = [] # Explain: Inverted pose output so I can run command line evo checks
    body_opti_HTMs = []
    def opti_tracked_body_to_my_body(T_head_to_world):
        return T.T_head_to_body @ np.linalg.inv(T_head_to_world)
    
    opti_body_poses, opti_body_velocities = aggregate_tracker(opti_tracked_body_to_my_body, np.array(opti_data))
    body_opti_HTMs = opti_body_poses
    # Each pose is the IMU pose in the optitrack world frame.

    # aggregate_tracker returns [timestamp, flattened HTM]
    # for opti_tum_traj we need to convert back to tum
    for pose in opti_body_poses: 
        body_opti_tum_traj.append(slam_HTM_to_TUM(pose))

        timestamp = pose[0]
        tum_vec = HTM_to_TUM(np.linalg.inv(pose[1:].reshape(4, 4) ))       # should return (7,) → [tx, ty, tz, qx, qy, qz, qw]
        result = np.concatenate(([timestamp], tum_vec))  # shape (8,)
        body_opti_inv_tum_traj.append(result)

    opti_json = [ {
            "t": float(body_pose[0]),
            "type": "opti_pose",
            "T_body_world" : body_pose[1:].reshape((4,4)),
        } for body_pose, body_v in zip( list(opti_body_poses), list(opti_body_velocities))]
    


    synth_uwb_json = []
    # synth_uwb_json = range_synthesizer2(START, END, body_opti_tum_traj, T, outpath)
    
    # Compose the final factor graph dataset
    all_data = uwb_json + imu_json + opti_json + synth_uwb_json
    for mes in all_data: mes["src"] = ID

    # Run sanity check to make sure measurements are at the frequency we expect them to be before testing in the graph

    print("Checking frequency of input data")
    print(f" Measured Synth UWB frequency {len(synth_uwb_json) / (END-START)}")
    print(f" Measured UWB frequency {len(uwb_json) / (END-START)}")
    print(f" Measured optitrack frequency {len(opti_data) / (END-START)}")

    # Filter to make sure all messages ( and data jsons ) fall within the ROS recording time interval, (because some of them don't apparently)
    all_data = filtt(all_data)
    all_data = sorted(all_data, key=lambda x: x["t"])

    return all_data, anchor_positions, T, body_opti_tum_traj
    

if __name__ == "__main__":
    # Example usage:
    # Single user
        # python3 post_process.py 2 opti_circle_test1 --opti --slam 20
        # keep opti at default frequency, subsample SLAM to 20Hz.
    # Multi user
        # python3 post_process.py 0 opti_multi_circle --opti --slam -a -f --multi_merge

    NO_SUBSAMPLE = -10
    parser = argparse.ArgumentParser(description="Stream collector")
    parser.add_argument("trial_name" , type=str)

    args = parser.parse_args()


    merged_all = []
    gt_trajectories = []
    
    for user in [2,3,4]: # Hard coded removing 3 for opti_multi1
        print(f"\nNUC{user}\n")

        args.id = user # Override whatever ID gets passed in
        all, anchor_positions, T, body_opti_tum_traj = post_process(args)
        merged_all = merged_all + all
        gt_trajectories.append(body_opti_tum_traj)

    # Mirror UWB ranges. This is something the Beluga firmware could report, but doesn't.
    # It only logs range on the report, not on the final, this means the responder compute the range, but never logs it.
    # This will effectively double our ranges.
    mirrored_uwb = []
    for j in merged_all:
        if j["type"] == "uwb":
            j_ = copy.deepcopy(j)
            temp_src = j["src"]
            j_["src"] = j["id"]
            j_["id"] = temp_src
            mirrored_uwb.append(j_)
    merged_all += mirrored_uwb

    merged_all = sorted(merged_all, key=lambda x: x["t"])

    # error_analysis(4, merged_all, anchor_positions, gt_trajectories, T)

    for id in [2,3,4]:
        annotated_uwb = generate_parham_data(id, merged_all, anchor_positions, gt_trajectories, T, args.trial_name)
        json.dump(annotated_uwb, open(f'./filter_dev_data/{id}/{args.trial_name}.json', 'w'), cls=NumpyEncoder, indent=1)