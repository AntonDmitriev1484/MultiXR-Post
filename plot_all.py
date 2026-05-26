import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def draw_axes(ax, T, length=0.1):
    """Draw coordinate axes from transformation matrix T."""
    H = np.linalg.inv(T)
    origin = (H @ np.array([0,0,0,1]))[:3]
    x_axis = (H @ np.array([1,0,0,1]))[:3]
    y_axis = (H @ np.array([0,1,0,1]))[:3]
    z_axis = (H @ np.array([0,0,1,1]))[:3]

    ax.quiver(*origin, *(x_axis-origin) * length, color='r')
    ax.quiver(*origin, *(y_axis-origin) * length, color='g')
    ax.quiver(*origin, *(z_axis-origin) * length, color='b')

import json
import numpy as np
import matplotlib.pyplot as plt

# -1 plot trajectory but not coordinate frames
# -2 don't plot trajectory

DONT_PLOT=-2

def plot_trial(
    id,
    trial_name,
    slam_stride=-2,
    opti_stride=-2,
    est_stride=-2,
    est_path=None,          # <-- added
    show_live_slam=False,
    run_config="",
    label_text="",
    anchors=False,
    transforms_json=None,
    calibration=False,
    show=True,
    ax=None
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------
    # File paths
    # ------------------------------
    all_json_path = (
        f"/home/antond2/Desktop/Research/MultiXR-Post/"
        f"{id}/post/{trial_name}_post/all.json"
    )

    anchors_path = (
        f"/home/antond2/Desktop/Research/MultiXR-Post/"
        f"{id}/post/{trial_name}_post/anchors.json"
    )

    transforms_path = (
        f"/home/antond2/Desktop/Research/MultiXR-Post/"
        f"{id}/post/{trial_name}_post/transforms.json"
    )

    # ------------------------------
    # Load required data
    # ------------------------------
    with open(all_json_path, "r") as f:
        all_data = json.load(f)

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    T_imu_to_body = np.array(transforms["T_imu_to_body"])

    # ------------------------------
    # Initialize pose containers
    # ------------------------------
    slam_poses = []
    lost_slam_poses = []
    opti_poses = []

    aligned_slam_poses = []
    lost_aligned_slam_poses = []

    aligned_slam_use = "aligned_slam_pose"
    if show_live_slam: aligned_slam_use = "aligned_live_slam_pose"

    # ------------------------------
    # Parse SLAM + OptiTrack
    # ------------------------------
    for item in all_data:
        if (
            item.get("type") in ("slam_pose")
            and "T_body_world" in item
        ):
            pose = np.array(item["T_body_world"])
            if item.get("status") == "lost":
                slam_poses.append(None)
                lost_slam_poses.append(pose)
            else:
                slam_poses.append(pose)
                lost_slam_poses.append(None)
        elif ( item.get("type") == aligned_slam_use):
            pose = np.array(item["T_body_world"])
            if item.get("status") == "lost":
                aligned_slam_poses.append(None)
                lost_aligned_slam_poses.append(pose)
            else:
                aligned_slam_poses.append(pose)
                lost_aligned_slam_poses.append(None)
        elif (
            item.get("type") == "opti_pose"
            and "T_body_world" in item
        ):
            opti_poses.append(np.array(item["T_body_world"]))

    # Aligned SLAM will automatically overrule regular SLAM in plotting
    if len(aligned_slam_poses) > 0:
        slam_poses = aligned_slam_poses
        lost_slam_poses = lost_aligned_slam_poses

    # ------------------------------
    # Create figure if needed
    # ------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # ------------------------------
    # Plot SLAM
    # ------------------------------
    if len(slam_poses) > 0 and slam_stride != DONT_PLOT:
        positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in slam_poses
        ])

        ax.plot(
            positions_world[:, 0],
            positions_world[:, 1],
            positions_world[:, 2],
            label="SLAM",
            color="green"
        )

        lost_positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in lost_slam_poses
        ])

        ax.plot(
            lost_positions_world[:, 0],
            lost_positions_world[:, 1],
            lost_positions_world[:, 2],
            label="LOST SLAM",
            color="red"
        )

        valid_slam = [p for p in slam_poses if p is not None]

        if slam_stride > 0:
            for i in range(0, len(valid_slam), slam_stride):
                draw_axes(ax, valid_slam[i], length=0.4)

    # ------------------------------
    # Plot OptiTrack
    # ------------------------------
    if len(opti_poses) > 0 and opti_stride != DONT_PLOT:
        positions_world = np.array([
            np.linalg.inv(p)[:3, 3]
            for p in opti_poses
        ])

        ax.plot(
            positions_world[:, 0],
            positions_world[:, 1],
            positions_world[:, 2],
            label="Optitrack",
            color="lightgreen"
        )

        if opti_stride > 0:
            for i in range(0, len(opti_poses), opti_stride):
                draw_axes(ax, opti_poses[i], length=0.4)

    # ------------------------------
    # Plot estimates (optional)
    # ------------------------------
    if est_path is not None:
        try:
            with open(est_path, "r") as f:
                est_data = json.load(f)

            est_poses = []

            for item in est_data:
                if (
                    item.get("type") == "est_pose"
                    and "T_body_world" in item
                ):
                    est_poses.append(
                        np.array(item["T_body_world"])
                    )

            if len(est_poses) > 0 and est_stride != DONT_PLOT:
                positions_world = np.array([
                    np.linalg.inv(p)[:3, 3]
                    for p in est_poses
                ])

                ax.scatter(
                    positions_world[:, 0],
                    positions_world[:, 1],
                    positions_world[:, 2],
                    label=label_text,
                    color="purple",
                    s=0.5
                )

                if est_stride > 0:
                    for i in range(0, len(est_poses), est_stride):
                        draw_axes(ax, est_poses[i], length=0.4)

        except Exception as e:
            print(f"Estimate plotting failed: {e}")

    # ------------------------------
    # Anchors
    # ------------------------------
    if anchors:
        try:
            with open(anchors_path, "r") as f:
                anchor_data = json.load(f)

            for d in anchor_data:
                pos = d["position"]

                ax.scatter(pos[0], pos[1], pos[2], color="purple")
                ax.text(pos[0], pos[1], pos[2], d["ID"])

        except Exception as e:
            print(f"Anchor plotting failed: {e}")

    # ------------------------------
    # Plot formatting
    # ------------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_title(f"NUC{id} {trial_name}")
    ax.view_init(elev=45, azim=45)
    ax.legend()

    if show:
        plt.show()

    return ax

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("id", type=int)
    parser.add_argument("trial_name")

    parser.add_argument(
        "--slam",
        type=int,
        nargs="?",
        const=-1,
        default=-2,
        help="Optional stride. If passed without value, uses 1. -1 disables."
    )
    parser.add_argument(
        "--opti",
        type=int,
        nargs="?",
        const=-1,
        default=-2,
        help="Optional stride. If passed without value, uses 1. -1 disables."
    )
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--transforms_json", default=None)
    parser.add_argument("--calibration", action="store_true")

    args = parser.parse_args()

    # -1 means don't plot coordinate axes on trajectory

    plot_trial(
        id=args.id,
        trial_name=args.trial_name,
        slam_stride=args.slam,
        opti_stride=args.opti,
        anchors=args.anchors,
        transforms_json=args.transforms_json,
        calibration=args.calibration,
    )


if __name__ == "__main__":
    main()