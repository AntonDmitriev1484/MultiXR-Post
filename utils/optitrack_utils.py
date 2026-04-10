
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.math_utils import * 

from utils.load_rostypes import *
from utils.ros_msg_handlers import *
from rosbags.highlevel import AnyReader

def load_optitrack(bagpath, id):

    typestore = load_rostypes()

    optitrack_poses = []

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in ["/tf"]]
        for connection, timestamp, rawdata in reader.messages(connections=connections):

            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                for tf in msg.transforms:
                    if tf.child_frame_id != f'Helmet{id}': continue
                    t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    tz = tf.transform.translation.z
                    qx = tf.transform.rotation.x
                    qy = tf.transform.rotation.y
                    qz = tf.transform.rotation.z
                    qw = tf.transform.rotation.w

                    pose = np.array([t, tx, ty, tz, qx, qy, qz, qw])
                    optitrack_poses.append(pose)
            except Exception:
                print( "Exception! skipped message")
                continue  # optionally log here

    return optitrack_poses


def euler_to_tum(arr, degrees=True):
    """
    Convert Euler + translation to TUM format [t tx ty tz qx qy qz qw].
    """
    rot = R.from_euler('xyz', arr[1:4], degrees=degrees)
    qx, qy, qz, qw = rot.as_quat()  # [x, y, z, w]
    return np.array([arr[0], arr[4], arr[5], arr[6], qx, qy, qz, qw])



# Crop all Vicon data to be within the ROS timestamps
# Assumption is that Vicon data timestamps are clock synced with NUC.
def crop_opti(opti_data, start, end):
    opti_data = [ d for d in opti_data if start < d[0] and d[0] < end ] # Doesn't mutate vicon data
    return opti_data


def clean_opti(opti_data):

    # If you're mobile and translation suddenly drop to 0, that means tracking was lost. interpolate that thang

    data = opti_data

    # In case we start off at a 0 pose, find the first non-zero pose
    # and set that to be our start pose

    def is_outlier(tum_pose):
        norm = np.linalg.norm(np.array(tum_pose)[1:])
        return norm <= 1e-5

    # If our starting pose is an outlier and we have nothing to interpolate between
    start_pose = None
    for i in range(0, len(data)):
        if not is_outlier(data[i]): 
            start_pose = np.array(data[i]) # Next valid TUM timestamped pose
            break

    for p in range(0,i):
        data[p] = start_pose

    # Now clean
    for i in range(1, len(data)):
        if is_outlier(data[i]):

            last_pose = np.array(data[i-1]) # Last valid TUM timestamped pose
            next_pose = None
            interp_pose = None
            for j in range(i+1, len(data)): # Find next valid TUM timestamped pose
                # print(data)
                if not is_outlier(data[j]): 
                    next_pose = np.array(data[j]) # Next valid TUM timestamped pose
                    current_timestamp = data[i][0]
                    interp_pose = interpolate_pose(
                        slam_quat_to_HTM(last_pose), last_pose[0],
                        slam_quat_to_HTM(next_pose), next_pose[0],
                        current_timestamp, 100
                    )
                    break

            if interp_pose is not None:
                interp_pose = HTM_to_TUM(interp_pose) # Returns a non timestamped HTM
                data[i] = np.insert(interp_pose, 0, current_timestamp) #I'm pretty sure this mutates the original array?

    opti_data = data
    return opti_data




def get_tx_position(T_vuwb_to_uwbtx, data):
    positions = []
    for pose in data: # Loop through until you find a pose that is not an outlier
        if np.linalg.norm(np.array(pose)[1:]) > 1e-5:
            T_vuwb_to_world = slam_quat_to_HTM(pose)
            T_world_to_tx = T_vuwb_to_uwbtx @ np.linalg.inv(T_vuwb_to_world)
            position = np.linalg.inv(T_world_to_tx)[:3,3]
            return position