import numpy as np
from utils.math_utils import *
import matplotlib.pyplot as plt
import json
import copy

# Body (IMU) trajectory in world frame
# Default UWB frequency is 5

# def range_synthesizer(START, END, body_traj, T, outpath, f_uwb=5/4.0, std=0.2):
#     anchors = {
#             1: [2, 0.5, 3],
#             3: [-6, 2, 2],
#             4: [-3, -3, 1],
#             5: [0, -4, 2]
#     }
#     # anchors = {
#     #         1: [2, 0.5, 3],
#     #         3: [-6, 2, 2],
#     #         4: [-3, -3, 1],
#     #         5: [0, -4, 2],
#     #         6: [-2, 2, 1.5]
#     # }

#     write_anchors = []
#     synth_ranges = []

#     # First transform body_traj to uwb_rx_traj

#     dt_uwb = 1/f_uwb
#     timestamps = np.arange(START, END, dt_uwb)

#     body_traj = np.array(body_traj)

#     for anchor_id, position in anchors.items():
#         N_ranges = timestamps.shape[0]
        
#         write_anchors.append({"ID":anchor_id, "position":position})

#         for i in range(0, timestamps.shape[0]):
#             t = timestamps[i]
            
#             vicon_timestamps = body_traj[:,0]
#             tdiffs = np.abs(vicon_timestamps - t)
#             idx = np.argmin(tdiffs) # Get closest vicon pose to this timestamp

#             T_world_to_body_tum = body_traj[idx]
#             T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
#             T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
#             T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

#             ## NOTE: Something is getting inverted here, either in pose reporting or in range calculation

#             dest_position = position
#             source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
#             # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

#             dist = np.linalg.norm(dest_position -  source_position)
#             synth_range =  np.random.normal(loc=dist, scale=std)

#             j = {
#                 "t":t,
#                 "type": "synth_uwb",
#                 "tag": "synth_for_user",
#                 "src": 2,
#                 "id": anchor_id,
#                 "pose": T_world_to_decawave, 
#                 "range": synth_range
#             }
#             synth_ranges.append(j)
        
#     out_anchors = open(f'{outpath}/anchors.json', 'w')
#     class NumpyEncoder(json.JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             if hasattr(obj, '__dict__'):
#                 return vars(obj)
#             return super().default(obj)
        
#     print(write_anchors)
#     json.dump(write_anchors, out_anchors, cls=NumpyEncoder, indent=1)

#     return synth_ranges



# Returns evenly spaced ranges
def range_synthesizer2(START, END, body_traj, T, outpath, f_uwb=5, std=0.01):
    anchors = {
            1: [2, 0.5, 3],
            3: [-6, 2, 2],
            2: [-3, -3, 1],
            5: [0, -4, 2]
    }
    # anchors = {
    #         1: [2, 0.5, 3],
    #         3: [-6, 2, 2],
    #         4: [-3, -3, 1],
    #         5: [0, -4, 2],
    #         6: [-2, 2, 1.5]
    # }

    write_anchors = []
    synth_ranges = []

    # First transform body_traj to uwb_rx_traj

    dt_uwb = 1/f_uwb
    timestamps = np.arange(START, END, dt_uwb)

    body_traj = np.array(body_traj)

    for n, (anchor_id, position) in enumerate(anchors.items()):
        N_ranges = timestamps.shape[0]

        shift = (n/len(anchors.items()))
        print(f"{anchor_id} shift = {shift}")
        this_timestamps = timestamps + shift # Each anchors timestamps are shifted over by
        
        write_anchors.append({"ID":anchor_id, "position":position})

        for i in range(0, this_timestamps.shape[0]):
            t = this_timestamps[i]
            
            vicon_timestamps = body_traj[:,0]
            tdiffs = np.abs(vicon_timestamps - t)
            idx = np.argmin(tdiffs) # Get closest vicon pose to this timestamp

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            ## NOTE: Something is getting inverted here, either in pose reporting or in range calculation

            dest_position = position
            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
            # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

            dist = np.linalg.norm(dest_position -  source_position)
            synth_range =  np.random.normal(loc=dist, scale=std)

            j = {
                "t":t,
                "type": "uwb",
                "tag": "synth_for_user",
                "src": 2,
                "id": anchor_id,
                "pose": T_world_to_decawave, 
                "range": synth_range
            }
            synth_ranges.append(j)
        
    out_anchors = open(f'{outpath}/anchors.json', 'w')
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, '__dict__'):
                return vars(obj)
            return super().default(obj)
        
    print(write_anchors)
    json.dump(write_anchors, out_anchors, cls=NumpyEncoder, indent=1)

    return synth_ranges


# Compute range error for inter-user and user-anchor ranges.
def error_analysis( id, multi_all, anchors, gt_trajectories, T, users):

    mobile_nodes = users
    # First transform body_traj to uwb_rx_traj

    range_errors = {}
    nlos_metric = {}
    timestamps = {}

    all_range_errors = []
    all_range_ts = []

    uwb = [j for j in multi_all if (j["src"] == id and j["type"] == "uwb")]

    for j in uwb:
        if j["id"] in mobile_nodes:

            other_body_traj = np.array(gt_trajectories[j["id"]-2])
            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            tx_position = T_decawave_to_world[:3,3]

            idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = other_body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            other_tx_position = T_decawave_to_world[:3,3]

            # Metrics recorded here
            synth_range = np.linalg.norm(tx_position -  other_tx_position)
            mes_range = j["range"]

            err = np.abs(synth_range - mes_range)
            range_errors.setdefault(j["id"], []).append(err)
            all_range_errors.append(err)
            all_range_ts.append(j["t"])

            timestamps.setdefault(j["id"], []).append(j["t"])

            A = 121.74

            fp_power_ = 10 * np.log10( ((j["firstpathamp1"] ** 2) + (j["firstpathamp2"]**2) + (j["firstpathamp3"]**2)) 
                                    / (j["rxpreamcount"]**2) ) - A
            rx_power_ = 10 * np.log10( j["maxgrowthcir"] * (2**17) / (j["rxpreamcount"] ** 2)) - A
            
            nlos_metric.setdefault(j["id"], []).append(rx_power_ - fp_power_)

        else:

            dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame

            # Metrics recorded here
            synth_range = np.linalg.norm(dest_position -  source_position)
            mes_range = j["range"]
            timestamps.setdefault(j["id"], []).append(j["t"])

            err = np.abs(synth_range - mes_range)
            range_errors.setdefault(j["id"], []).append(err)
            all_range_errors.append(err)
            all_range_ts.append(j["t"])

            A = 121.74

            fp_power_ = 10 * np.log10( ((j["firstpathamp1"] ** 2) + (j["firstpathamp2"]**2) + (j["firstpathamp3"]**2)) 
                                    / (j["rxpreamcount"]**2) ) - A
            rx_power_ = 10 * np.log10( j["maxgrowthcir"] * (2**17) / (j["rxpreamcount"] ** 2)) - A

            nlos_metric.setdefault(j["id"], []).append(rx_power_ - fp_power_)


    print()
    for node, err in range_errors.items():

        if node == id:
            continue

        nlos_score = nlos_metric[node]
        t = timestamps[node]
        # print(timestamps)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # print(err)
        axs[0].plot(t, err)
        axs[0].set_title(f"Range Errors from {id} to {node}")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Error (m)")
        axs[0].grid(True)


        axs[1].plot(t, nlos_score)

        axs[1].set_title("NLOS Metric")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("NLOS score")
        axs[1].grid(True)

        plt.tight_layout()

    plt.show()


# Compute range error for inter-user and user-anchor ranges.
def generate_parham_data( id, multi_all, anchors, gt_trajectories, T, trial_name, users):

    
    mobile_nodes = users
    # First transform body_traj to uwb_rx_traj

    range_errors = {}
    nlos_metric = {}
    timestamps = {}

    all_range_errors = []
    all_range_ts = []


    uwb = [j for j in multi_all if (j["src"] == id and j["type"] == "uwb")]
    annotated_uwb = []

    for j in uwb:
        if j["id"] in mobile_nodes:

            # print(len(gt_trajectories))
            print(f"{j['id']-2}=")
            other_body_traj = np.array(gt_trajectories[j["id"]-2])
            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            tx_position = T_decawave_to_world[:3,3]

            idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = other_body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            other_tx_position = T_decawave_to_world[:3,3]

            annotated_range = copy.deepcopy(j)
            annotated_range["optitrack_src_tx_position"] = tx_position
            annotated_range["optitrack_dst_tx_position"] = other_tx_position
            annotated_uwb.append(annotated_range)

        else:

            dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame

            # Metrics recorded here
            annotated_range = copy.deepcopy(j)
            annotated_range["optitrack_src_tx_position"] = source_position
            annotated_range["optitrack_dst_tx_position"] = dest_position
            annotated_uwb.append(annotated_range)

    return annotated_uwb

# # Input trajectories are T_world_to_body we need T_body_world
def range_synthesizer3( multi_all, anchors, gt_trajectories, T, std):

    # Replace all real ranges with synthetic.

    mobile_nodes = [2,3,4]
    # First transform body_traj to uwb_rx_traj

    rs = [j["range"] for j in multi_all if j["type"] == "uwb"]
    print(f"{rs[:5]=}")



    for id in mobile_nodes: # For each user
        for j in multi_all:
            if j["src"] == id and j["type"] == "uwb": # Find all ranges that belong to that user, and convert them to synthetic.

                if j["id"] in mobile_nodes: # If we are ranging to another user

                    other_body_traj = np.array(gt_trajectories[j["id"]-2])
                    body_traj = np.array(gt_trajectories[j["src"]-2])

                    idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
                    T_world_to_body_tum = body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    tx_position = T_decawave_to_world[:3,3]

                    idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
                    T_world_to_body_tum = other_body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    other_tx_position = T_decawave_to_world[:3,3]

                    # synth_range = np.random.normal(loc=np.linalg.norm(tx_position -  other_tx_position), scale=std)
                    synth_range = np.linalg.norm(tx_position -  other_tx_position)

                    j["range"] = synth_range


                else: # If we are ranging to an anchor

                    dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

                    t = j["t"]
                    body_traj = np.array(gt_trajectories[j["src"]-2])

                    idx = np.argmin(np.abs(body_traj[:, 0] - t))

                    T_world_to_body_tum = body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

                    source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
                    # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

                    # synth_range = np.random.normal(loc=np.linalg.norm(source_position -  dest_position), scale=std)
                    synth_range = np.linalg.norm(source_position -  dest_position)

                    j["range"] = synth_range

    rs = [j["range"] for j in multi_all if j["type"] == "uwb"]
    print(f"{rs[:5]=}")

    return multi_all


from copy import deepcopy
import numpy as np


from copy import deepcopy
import numpy as np


# Input trajectories are T_world_to_body, we need T_body_to_world
def range_synthesizer4(
    multi_all,
    anchors,
    gt_trajectories,
    T,
    std=0.2,
    rate=10,
):

    mobile_nodes = [2, 3, 4]

    def compute_range(src, dst, t):

        body_traj = np.array(gt_trajectories[src - 2])

        idx = np.argmin(np.abs(body_traj[:, 0] - t))

        T_world_to_body_tum = body_traj[idx]
        T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)

        T_decawave_to_world = (
            np.linalg.inv(T_world_to_body)
            @ np.linalg.inv(T.T_body_to_decawave)
        )

        src_pos = T_decawave_to_world[:3, 3]

        # Mobile -> Mobile
        if dst in mobile_nodes:

            other_body_traj = np.array(
                gt_trajectories[dst - 2]
            )

            idx = np.argmin(
                np.abs(other_body_traj[:, 0] - t)
            )

            T_world_to_body_tum = other_body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(
                T_world_to_body_tum
            )

            T_decawave_to_world = (
                np.linalg.inv(T_world_to_body)
                @ np.linalg.inv(T.T_body_to_decawave)
            )

            dst_pos = T_decawave_to_world[:3, 3]

        # Mobile -> Anchor
        else:

            dst_pos = np.array(
                [
                    x["position"]
                    for x in anchors
                    if x["ID"] == dst
                ][0]
            )

        true_range = np.linalg.norm(
            src_pos - dst_pos
        )

        return np.random.normal(
            true_range,
            std
        )

    #
    # Gather all existing UWB links
    #
    src_groups = {}

    for msg in multi_all:

        if msg.get("type") != "uwb":
            continue

        if msg["src"] not in mobile_nodes:
            continue

        src_groups.setdefault(msg["src"], {})
        src_groups[msg["src"]].setdefault(
            msg["id"],
            []
        )
        src_groups[msg["src"]][msg["id"]].append(msg)

    #
    # Keep non-UWB messages
    #
    output = [
        m
        for m in multi_all
        if m.get("type") != "uwb"
    ]

    #
    # Generate synthetic schedule
    #
    dt = 1.0 / rate

    for src, dst_groups in src_groups.items():

        #
        # Determine overall time span
        #
        all_msgs = []

        for msgs in dst_groups.values():
            all_msgs.extend(msgs)

        all_msgs.sort(key=lambda x: x["t"])

        if len(all_msgs) == 0:
            continue

        start_t = all_msgs[0]["t"]
        end_t = all_msgs[-1]["t"]

        #
        # Destinations this source can range to
        #
        dst_ids = sorted(dst_groups.keys())

        #
        # Use first measurement on each link
        # as a template
        #
        templates = {
            dst: deepcopy(dst_groups[dst][0])
            for dst in dst_ids
        }

        desired_times = np.arange(
            start_t,
            end_t,
            dt
        )

        #
        # Round-robin schedule
        #
        for k, t in enumerate(desired_times):

            dst = dst_ids[
                k % len(dst_ids)
            ]

            msg = deepcopy(
                templates[dst]
            )

            msg["t"] = float(t)

            msg["range"] = compute_range(
                src,
                dst,
                t
            )

            output.append(msg)

    output.sort(
        key=lambda x: x.get("t", 0.0)
    )

    return output