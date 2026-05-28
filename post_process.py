
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
from utils.ros1_slam_utils import *


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return super().default(obj)

### Process optitrack data before alignment
def prep_optitrack_output(args, START, END, T, opti_data):
    opti_json = []
    body_opti_tum_traj = [] # Explain: Used later in UWB error code
    body_opti_HTMs = [] # Explain: Used later in Umeyama alignment
    if args.opti is not None:
        def opti_tracked_body_to_my_body(T_head_to_world):
            return T.T_head_to_body @ np.linalg.inv(T_head_to_world)
        
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
            } for body_pose, body_v in zip( list(opti_body_poses), list(opti_body_velocities))]
        
    return opti_json, body_opti_HTMs, body_opti_tum_traj

### Process SLAM data before alignment
def prep_orbslam3_output(args, START, END, T, post_slam_data, live_slam_data=[], real_failures=False):
    post_slam_json = []
    live_slam_json = []
    body_post_slam_HTMs = [] # Used later in Umeyama alignment
    body_live_slam_HTMs = [] # Used later in Umeyama alginment, if real failures.

    print(f"{real_failures=}")

    if args.slam is not None:
        post_slam_data[:,0] *= 1e-9 # Adjust timestamps to be in 's'

        def slam_tracked_body_to_my_body(T_cam1_to_sorigin): # SLAM quat gives you the transform from cam1 frame to slam origin
            return T.T_cam1_to_body @ np.linalg.inv(T_cam1_to_sorigin)

        slam_body_poses, slam_body_velocities = aggregate_tracker(slam_tracked_body_to_my_body, post_slam_data)
        body_post_slam_HTMs = slam_body_poses
        # Each pose is the IMU pose in the slam frame. i.e. T_slamworld_to_imu. This is what plotting assumes.

        # If we passed a valid frequency to subsample to
        slam_freq = len(post_slam_data) / (END-START)
        if slam_freq > args.slam > 0:
            skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
            slam_body_poses = np.array(slam_body_poses)
            slam_body_poses = slam_body_poses[::skip] # Finally, subsample to required frequency

        # Convert from nparray to json format
        post_slam_json = [ {
                "t": float(body_pose[0]),
                "type": "slam_pose",
                "T_body_world" : body_pose[1:].reshape((4,4)),

            } for body_pose, body_v in zip( list(slam_body_poses), list(slam_body_velocities))]
        
        if real_failures:
            live_slam_data[:,0] *= 1e-9 # Adjust timestamps to be in 's'
            live_slam_body_poses, live_slam_body_velocities = aggregate_tracker(slam_tracked_body_to_my_body, live_slam_data)
            body_live_slam_HTMs = live_slam_body_poses
            # IF THERES A REAL FAILURE, OVERWRITE unaligned save_traj SLAM with live SLAM
            live_slam_json = [ {
                    "t": float(body_pose[0]),
                    "type": "slam_pose",
                    "T_body_world" : body_pose[1:].reshape((4,4)),

                } for body_pose, body_v in zip( list(live_slam_body_poses), list(live_slam_body_velocities))]

    return post_slam_json, live_slam_json, body_post_slam_HTMs, body_live_slam_HTMs

### Just annotate
def prep_annotate_slam_output(args, START, END, T, aligned_slam_data, mes_type, real_failures=None):

    real_failures = real_failures[0]

    def identity(T_body_to_world): # Already in body frame.
        return T_body_to_world
    print(aligned_slam_data.shape)
    # Aggregate is basically just for format conversion
    aligned_slam_body_poses, aligned_slam_body_velocities = aggregate_tracker(identity, np.array(aligned_slam_data))
    # Each pose is the IMU pose in the optitrack world frame.
    print(aligned_slam_body_poses.shape)

    fail_start_t = START + real_failures["start"] # Timestamp failure starts and we rely on IMU
    new_map_start_t = START + real_failures["init_newmap"] # Timestamp that we start tracking a new map
    relocalized_t = START + real_failures["end"] # Timestamp we end failure
    imu_only_idxs = np.where( 
        (new_map_start_t >= aligned_slam_body_poses[:,0]) & 
        (aligned_slam_body_poses[:,0]>= fail_start_t)
        )[0]
    newmap_idxs = np.where( 
        (relocalized_t >= aligned_slam_body_poses[:,0]) &
        (aligned_slam_body_poses[:,0] >= new_map_start_t)
        )[0]

    # Also do all of the labeling in here of tracking vs no tracking

    aligned_live_slam_json = [ {
            "t": float(body_pose[0]),
            "type": mes_type,
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": float(body_v[1]),
                    "vy": float(body_v[2]),
                    "vz": float(body_v[3])
            }
        } for body_pose, body_v in zip( list(aligned_slam_body_poses), list(aligned_slam_body_velocities))]
    
    # Apply tracking status labels
    for i in range(len(aligned_live_slam_json)):
        if i in imu_only_idxs: aligned_live_slam_json[i]["status"] = "imu"
        elif i in newmap_idxs: aligned_live_slam_json[i]["status"] = "newmap"
        else: aligned_live_slam_json[i]["status"] = "tracking"

    return aligned_live_slam_json

### Annotate and transform
def prep_synth_fail_slam_output(args, START, END, T, aligned_slam_data, mes_type, synth_failures=None):

    def identity(T_body_to_world): # Already in body frame.
        return T_body_to_world
    # Aggregate is basically just for format conversion
    aligned_slam_body_poses, aligned_slam_body_velocities = aggregate_tracker(identity, np.array(aligned_slam_data))

    imu_only_idx = []
    newmap_idxs = []
    if synth_failures is not None:

        synth_failures = synth_failures[0] # For now just assuming 1 failure
        fail_start_t = START + synth_failures["start"] # Timestamp failure starts and we rely on IMU
        new_map_start_t = START + synth_failures["init_newmap"] # Timestamp that we start tracking a new map
        relocalized_t = START + synth_failures["end"] # Timestamp we end failure

        imu_only_idxs = np.where( (new_map_start_t >= aligned_slam_body_poses[:,0]) & (aligned_slam_body_poses[:,0]>= fail_start_t))[0]
        newmap_idxs = np.where( (relocalized_t >= aligned_slam_body_poses[:,0]) & (aligned_slam_body_poses[:,0] >= new_map_start_t))[0]
        # Create the newmap segment that starts at the origin
        # newmap_idxs = np.where( relocalized_t >= aligned_slam_body_poses[:,0] >= new_map_start_t)
        
        traj_origin = aligned_slam_body_poses[0,1:].reshape((4,4)).copy()
        new_map_origin = aligned_slam_body_poses[newmap_idxs[0],1:].reshape((4,4)).copy() # This is world to body
        for i in newmap_idxs:
            body_pose = aligned_slam_body_poses[i,1:].reshape((4,4)).copy() # This is world to body
            new_map_pose =  body_pose @ np.linalg.inv(new_map_origin) @ traj_origin
            aligned_slam_body_poses[i,1:] = new_map_pose.flatten()

    # Also do all of the labeling in here of tracking vs no tracking

    aligned_post_slam_json = [ {
            "t": float(body_pose[0]),
            "type": mes_type,
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": float(body_v[1]),
                    "vy": float(body_v[2]),
                    "vz": float(body_v[3])
            }
        } for body_pose, body_v in zip( list(aligned_slam_body_poses), list(aligned_slam_body_velocities))]
    
    # Apply tracking status labels
    for i in range(len(aligned_post_slam_json)):
        if i in imu_only_idxs: aligned_post_slam_json[i]["status"] = "imu"
        elif i in newmap_idxs: aligned_post_slam_json[i]["status"] = "newmap"
        else: aligned_post_slam_json[i]["status"] = "tracking"

    body_slam_aligned_tum_traj = [HTM_to_TUM(p) for p in aligned_slam_body_poses]

    return aligned_post_slam_json, body_slam_aligned_tum_traj

def post_process(args):

    ID = args.id
    outpath = f'./{ID}/post/{args.trial_name}_post'
    out_world = outpath+f'/world/' # Vicon can define apriltags and anchors set up in world frame
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(out_world, exist_ok=True)

    in_slam = ""
    post_slam_data = []
    in_real_fails = f'./real_fails/'
    real_failures = f"{args.trial_name}_nuc{args.id}_slam_cam_traj.csv" in os.listdir(in_real_fails)

    # SLAM 'save_traj' trajectory
    in_slam = f'./{ID}/orbslam/out/{args.trial_name}_nuc{ID}_raw_cam_traj.txt'
    post_slam_data = np.loadtxt(in_slam)

    # SLAM 'ros1 bag trajectory'
    live_slam_data = []
    if real_failures:
        print("REAL SLAM FAILURE CASE, reading live trajectory frp ros1 bag csv")
        live_slam_data = parse_ros1_bag_csv(f'./real_fails/{args.trial_name}_nuc{args.id}_slam_cam_traj.csv')

    inpath = f'./{ID}/collect/{args.trial_name}_nuc{ID}_raw'
    in_kalibr_dir = inpath+f"/calibration/"
    kalibr_files = list(Path(in_kalibr_dir).glob("*.yaml"))
    in_kalibr = f"{kalibr_files[0]}"

    imu_json = json.load(open(inpath+'/imu_raw.json', 'r'))
    uwb_json = json.load(open(inpath+'/uwb_raw.json', 'r'))
    metadata = json.load(open(inpath+'/meta.json', 'r'))


    # # Filter for messages within bag timestamp range.
    # START = metadata["start_ns"] * 1e-9

    START = post_slam_data[0,0] * 1e-9 # Adjust timestamps to be in 's'

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
    T = define_transforms(in_kalibr)

    # Position of each UWB anchor transmitter in the optitrack world frame
    anchor_positions = {}
    if args.opti is not None:
        anchor_positions = compute_anchors(opti_anchor_trajectories, T.T_optiuwb_to_uwbtx)

    ### Process SLAM data
    # All trajectories have a post SLAM, and if they have real failures, they will have a live SLAM as well.
    # Convert to body frame, subsample, 
    # output body in SLAM world frame trajectory as jsons, and HTMs (for later alignment)
    post_slam_json, live_slam_json, body_post_slam_HTMs, body_live_slam_HTMs = (
        prep_orbslam3_output(args, START, END, T, post_slam_data, live_slam_data, real_failures)
        if args.slam is not None
        else ([], [], [], [])
    )

    ### Process Optitrack data
    # Convert to body frame, subsample, 
    # output body in optitrack world frame trajectory as jsons, and HTMs (for later alignment)
    # ouptut as a TUM trajectory, for later UWB error calculation code
    opti_json, body_opti_HTMs, body_opti_tum_traj = (
        prep_optitrack_output(args, START, END, T, opti_data)
        if args.opti is not None
        else ([], [], [])
    )

    ### Align SLAM trajectories to optitrack shared frame
    aligned_post_slam_json = []
    aligned_live_slam_json = []
    # Get HTMs here
    body_post_slam_aligned_tum_traj = []
    body_live_slam_aligned_tum_traj = []
    if args.align:

        if real_failures:
            # Compute alignment using the SLAM 'save_traj' trajectory to the Optitrack trajectory
            # apply alignment to SLAM 'ros1 bag' trajectory

            real_failures = json.load(open(f'./real_fails/{args.trial_name}_nuc{args.id}_fail.json','r'))
            all_data_start_ts = metadata["start_ns"] * 1e-9
            fail_end_ts = real_failures[0]["end"] + all_data_start_ts      
            # output body live slam trajectory - for use in evaluation
            body_post_slam_aligned_tum_traj, body_live_slam_aligned_tum_traj = bonus_umeyama_alignment(body_opti_HTMs, body_post_slam_HTMs, body_live_slam_HTMs, fail_end_ts)

            # annotate the live slam trajectory with real failures file - for use in plotting
            aligned_live_slam_json = prep_annotate_slam_output(args, all_data_start_ts, END, T, np.array(body_live_slam_aligned_tum_traj), "aligned_live_slam_pose", real_failures)

            # annotate the post slam trajectory in the same way - for use in graph
            aligned_post_slam_json = prep_annotate_slam_output(args, all_data_start_ts, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_slam_pose", real_failures)
            
        elif args.synth_failures: 
            # For now lets just assume that I will never add a synthetic failure to a real failure trajectory
            # otherwise the synthetic will overwrite the real failure annotations and it'll make a mess.

            synth_failures = []
            try:
                synth_failures = json.load(open(f'./{ID}/synth_failures/{args.trial_name}.json','r'))
            except Exception as err:
                print("No synth failures detected, setting synth_failures = []")

            body_post_slam_aligned_tum_traj = umeyama_alignment1(body_opti_HTMs, body_post_slam_HTMs)

            # annotate the post slam trajectory, for use in the graph
            aligned_post_slam_json = prep_annotate_slam_output(args, START, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_slam_pose", synth_failures)

            # add newmap deformation and IMU segment, 
            # and annotate live slam trajectory for use in plotting (aligned_live_slam_json) and evaluation (body_live_slam_aligned_tum_traj)
            aligned_live_slam_json, body_live_slam_aligned_tum_traj = prep_synth_fail_slam_output(args, START, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_live_slam_pose", synth_failures)



    synth_uwb_json = []
    
    # Compose the final factor graph dataset
    all_data = uwb_json + imu_json + opti_json + post_slam_json + synth_uwb_json + aligned_post_slam_json + aligned_live_slam_json
    for mes in all_data: mes["src"] = ID


    ### Hard coding things for evaluation
    # Remove Node 2 from being used in multi2_board_loss3, it has un-controlled tracking loss.
    if args.trial_name == "multi2_board_loss3":
        for j in all_data:
            if j["src"] == 2:
                if j["type"] == "aligned_slam_pose": j["status"] = "lost"
                if j["type"] == "aligned_live_slam_pose": j["status"] = "lost"
                if j["type"] == "slam_pose": j["status"] = "lost" 

    # Remove Node 5 from being used in multi2_.._nlos trials. Optitrack didn't track it properly.
    if args.trial_name == "multi2_human_nlos" or args.trial_name == "multi2_object_nlos":
        copy_all_data = []
        for j in all_data:
            if (j["type"] == "uwb"): 
                if not (j["id"] == 5): copy_all_data.append(j)
            else: copy_all_data.append(j)
        all_data = copy_all_data

    ### Copy all world information: T, anchors, apriltags, to output

    if args.opti is not None:
        out_anchors = open(f'{outpath}/anchors.json', 'w')
        json.dump(anchor_positions, out_anchors, cls=NumpyEncoder, indent=1)


    with open(f'{outpath}/transforms.json', 'w') as fs: json.dump(vars(T), fs, cls=NumpyEncoder, indent=1)

    # Run sanity check to make sure measurements are at the frequency we expect them to be before testing in the graph

    print("Checking frequency of input data")
    print(f" Measured Synth UWB frequency {len(synth_uwb_json) / (END-START)}")
    print(f" Measured UWB frequency {len(uwb_json) / (END-START)}")
    if args.opti: print(f" Measured optitrack frequency {len(opti_data) / (END-START)}")
    print(f" Measured SLAM frequency {len(post_slam_data) / (END-START)}")

    print("Checking frequency of subsampled output")
    if args.opti: print(f" Measured subsampled optitrack frequency {len(opti_json) / (END-START)}")
    print(f" Measured subsampled SLAM frequency {len(post_slam_json) / (END-START)}")


    # Filter to make sure all messages ( and data jsons ) fall within the ROS recording time interval, (because some of them don't apparently)
    all_data = filtt(all_data)
    all_data = sorted(all_data, key=lambda x: x["t"])

    json.dump(all_data, open(outpath+"/all.json", 'w'), cls=NumpyEncoder, indent=1)

    # Write body in optitrack trajectory  to a csv file for loading into EVO
    np.savetxt(f"{outpath}/opti.txt", np.array(body_opti_tum_traj), fmt="%.8f")
    np.savetxt(f"{outpath}/aligned_slam.txt", np.array(body_post_slam_aligned_tum_traj), fmt="%.8f")
    np.savetxt(f"{outpath}/aligned_live_slam.txt", np.array(body_live_slam_aligned_tum_traj), fmt="%.8f")  
    # np.savetxt(f"{outpath}/opti_inv.txt", np.array(body_opti_inv_tum_traj), fmt="%.8f")
    # np.savetxt(f"{outpath}/slam_inv.txt", np.array(body_slam_inv_tum_traj), fmt="%.8f")

    if args.opti is None: 
        return all_data, {}, T, body_opti_tum_traj
    else:
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
    parser.add_argument("id", type=int)
    parser.add_argument("trial_name" , type=str)
    parser.add_argument("--opti", nargs="?", const=0, type=float)
    parser.add_argument("--slam", nargs="?", const=0, type=float)
    # Replace SLAM tracked body in SLAM frame trajectory, with SLAM tracked body in optitrack frame trajectory
    parser.add_argument("--align", "-a", action="store_true")
    # Load a json file containing tracking lost segments
    parser.add_argument("--synth_failures", "-f", action="store_true")

    parser.add_argument("--multi_merge", action="store_true")

    parser.add_argument("--check_uwb_error", action="store_true")

    args = parser.parse_args()

    if args.multi_merge:

        outpath = f'./merged/{args.trial_name}_merged/'
        os.makedirs(outpath, exist_ok=True)

        merged_all = []
        gt_trajectories = []
        
        for user in [2,3,4]:
            print(f"\nNUC{user}\n")

            args.id = user # Override whatever ID gets passed in
            all, anchor_positions, T, body_opti_tum_traj = post_process(args)
            merged_all = merged_all + all
            gt_trajectories.append(body_opti_tum_traj)
            # Record anchors
            if user==2: json.dump(anchor_positions, open(outpath+"/anchors.json", 'w'), cls=NumpyEncoder, indent=1)

            # Record transforms
            json.dump(vars(T), open(f'{outpath}/transforms{user}.json', 'w'), cls=NumpyEncoder, indent=1)
            
        
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

        # Synthetsize ranges in place of real ones.

        merged_all = sorted(merged_all, key=lambda x: x["t"])
        json.dump(merged_all, open(outpath+"/all.json", 'w'), cls=NumpyEncoder, indent=1)

        # UWB error analysis
        if args.check_uwb_error:
            error_analysis(4, merged_all, anchor_positions, gt_trajectories, T, users=[2,3,4])


    else:
        post_process(args)