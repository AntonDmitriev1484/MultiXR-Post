import numpy as np
from utils.math_utils import *
import json

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


# # TODO: For each real range, that occurs and we could "overhear" (we dont keep track of this) add in a synthetic range
# #some math first?
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
def range_synthesizer2(START, END, body_traj, T, outpath, f_uwb=5, std=0.2):
    anchors = {
            1: [2, 0.5, 3],
            3: [-6, 2, 2],
            4: [-3, -3, 1],
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