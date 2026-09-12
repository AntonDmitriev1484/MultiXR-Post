import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Okabe-Ito colorblind-safe palette
COLORS = {
    "opti": "#0072B2",       # blue
    "est": "#D55E00",  # vermillion
    "recovered": "#009E73",       # bluish green
    "anchor": "#CC79A7",        # reddish purple
    "lost": "#CC79A7",     # orange
}


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


# -1 plot trajectory but not coordinate frames
# -2 don't plot trajectory

DONT_PLOT=-2


def plot_trial(
    id,
    trial_name,
    slam_stride=-2,
    paths=None,
    fixed_lims=False,
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

    live_slam_path = all_json_path

    # Preserve function argument unless overridden
    if paths is not None:
        live_slam_path = paths.live_slam_path

    # ------------------------------
    # Initialize pose containers
    # ------------------------------

    # Load in local frame live slam poses
    slam_poses = []
    lost_slam_poses = []

    tracking = True
    with open(live_slam_path, "r") as f:
        for item in json.load(f):
            if (
                item.get("type") == "localframe_live_slam_pose"
                and "T_body_world" in item
            ):
                pose = np.array(item["T_body_world"])
                if item.get("status") == "tracking":
                    slam_poses.append(pose)
                    if not tracking: lost_slam_poses.append(pose) # Ensure continuity between red lost line and the visual recovered trajectory
                    lost_slam_poses.append(None)
                    tracking = True
                else:
                    slam_poses.append(None)
                    lost_slam_poses.append(pose)
                    tracking = False

    # ------------------------------
    # Create figure if needed
    # ------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # ------------------------------
    # Plot local frame live SLAM
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
            label="Local Frame Live SLAM",
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
    else:
        print("No localframe_live_slam_pose entries found")

    # ------------------------------
    # Plot formatting
    # ------------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # The local frame is the SLAM world frame, so the optitrack-frame box from
    # plot_all.py only makes sense when explicitly asked for.
    if fixed_lims:
        ax.set_xlim(-5, 1)
        ax.set_ylim(-2.5, 4)
        ax.set_zlim(-2, 2)

    ax.set_title(f"NUC{id} {trial_name}\nLocal Frame Live SLAM")

    ax.view_init(elev=45, azim=45)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("id", type=int)
    parser.add_argument("trial_name")

    parser.add_argument(
        "--slam",
        type=int,
        nargs="?",
        const=-1,
        default=-1,
        help="Optional stride. If passed without value, plots trajectory only. -2 disables."
    )
    parser.add_argument(
        "--fixed_lims",
        action="store_true",
        help="Use plot_all.py's hard-coded optitrack-frame axis limits."
    )

    args = parser.parse_args()

    # -1 means don't plot coordinate axes on trajectory

    plot_trial(
        id=args.id,
        trial_name=args.trial_name,
        slam_stride=args.slam,
        fixed_lims=args.fixed_lims,
    )


if __name__ == "__main__":
    main()
