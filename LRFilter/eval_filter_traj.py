#!/usr/bin/env python3
r"""
eval_filter_traj.py -- compare the SLAM trajectory produced from the *filtered*
frame set against the one produced from the *original* (raw) frame set.

Takes the name of a filtered-frame folder, e.g.

    opti_multi1_free_nuc3_recall100_K1
    \_________________/\____________/
        <trial>            filter tag

The trial name is everything up to and including the "nuc<N>" token, and <id> is
that N.  ORB-SLAM3 writes both trajectories to

    <WORKSPACE>/<id>/orbslam/out/<filtered_name>_filt_cam_traj.txt   (estimate)
    <WORKSPACE>/<id>/orbslam/out/<trial>_raw_cam_traj.txt            (reference)

They are TUM-format (t x y z qx qy qz qw) but ORB-SLAM3 writes the timestamps in
NANOSECONDS, which evo's TUM reader would happily load as "seconds" and then fail
to associate -- so the unit is detected and normalised here before anything is
handed to evo.

Reports translational and rotational APE to stdout and writes the same report to
<filtered_name>_eval.txt in the LRFilter directory (alongside this script), so
evaluations of different filter settings collect in one place.

    ./eval_filter_traj.py opti_multi1_free_nuc3_recall100_K1
    ./eval_filter_traj.py opti_multi1_free_nuc3_recall100_K1 --align-scale --plot
"""

import argparse
import os
import re
import sys

import numpy as np

FILTERED_NAME = "opti_multi1_free_nuc3_recall100_K1"

# Reports and plots are written here (the LRFilter directory), not next to the
# trajectories, so every filter evaluation collects in one place.
LRFILTER_DIR = os.path.dirname(os.path.abspath(__file__))

WORKSPACE = "/home/antond2/Desktop/Research/MultiXR-Post"
SLAM_OUTDIR = "{workspace}/{id}/orbslam/out"
SLAM_OUTPATH = SLAM_OUTDIR + "/{name}_{traj_kind}_traj.txt"

# ORB-SLAM3 timestamps are nanoseconds; these bounds identify the unit from the
# magnitude of a unix-epoch stamp (~1.78e9 s as of these trials).
_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def die(msg):
    sys.stderr.write("error: " + msg + "\n")
    sys.exit(1)


def parse_trial_and_id(filtered_name):
    """opti_multi1_free_nuc3_recall100_K1 -> ('opti_multi1_free_nuc3', '3').

    The trial name is the prefix up to and including the 'nuc<N>' token; whatever
    the filter appended after it (recall100_K1, ...) is the filter tag.
    """
    m = re.search(r"nuc(\d+)", filtered_name)
    if not m:
        die("could not parse a trial name out of %r -- expected a 'nuc<N>' token, "
            "e.g. opti_multi1_free_nuc3_recall100_K1; pass --trial/--id explicitly"
            % filtered_name)
    return filtered_name[:m.end()], m.group(1)


def filter_label(filtered_name, trial):
    """Legend label for the filtered run, e.g. 'LRFilter (Recall=95, K=2)'.

    Read from the filter tag so the figure cannot end up labelled with settings
    it was not produced from.
    """
    m = re.search(r"recall(\d+)_K(\d+)", filtered_name, re.IGNORECASE)
    if m:
        return "LRFilter (Recall=%s, K=%s)" % (m.group(1), m.group(2))
    tag = filtered_name[len(trial):].lstrip("_")
    return "LRFilter (%s)" % tag if tag else "LRFilter"


def detect_time_unit(stamps):
    """Guess the unit of unix-epoch timestamps from their magnitude."""
    t = float(np.median(np.abs(stamps)))
    if t > 1e17:
        return "ns"
    if t > 1e14:
        return "us"
    if t > 1e11:
        return "ms"
    return "s"


def load_tum(path, time_unit="auto"):
    """Read a TUM file into an evo PoseTrajectory3D, normalising time to seconds."""
    from evo.core.trajectory import PoseTrajectory3D

    if not os.path.isfile(path):
        die("trajectory file not found: %s" % path)

    raw = np.loadtxt(path, comments="#", ndmin=2)
    if raw.size == 0:
        die("trajectory file is empty: %s" % path)
    if raw.shape[1] != 8:
        die("%s has %d columns, expected 8 (t x y z qx qy qz qw)"
            % (path, raw.shape[1]))

    stamps = raw[:, 0]
    unit = detect_time_unit(stamps) if time_unit == "auto" else time_unit
    stamps = stamps * _UNIT_SCALE[unit]

    xyz = raw[:, 1:4]
    # TUM stores qx qy qz qw; evo wants w-first.
    quat_wxyz = raw[:, [7, 4, 5, 6]]
    norms = np.linalg.norm(quat_wxyz, axis=1)
    if np.any(norms < 1e-6):
        die("%s contains a degenerate (zero-norm) quaternion" % path)
    quat_wxyz = quat_wxyz / norms[:, None]

    order = np.argsort(stamps, kind="stable")
    traj = PoseTrajectory3D(positions_xyz=xyz[order],
                            orientations_quat_wxyz=quat_wxyz[order],
                            timestamps=stamps[order])
    return traj, unit


class Report(object):
    """Collects the report lines so stdout and <filtered_name>_eval.txt agree."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        self.lines.append(line)
        print(line)

    def describe(self, label, traj, path, unit):
        self("%-11s %s" % (label + ":", path))
        self("%-11s %d poses | %.2f s | %.3f m path | timestamps read as %s"
             % ("", traj.num_poses, traj.timestamps[-1] - traj.timestamps[0],
                traj.path_length, unit))

    def stats(self, title, stats, unit):
        self("")
        self("  " + title)
        self("  " + "-" * len(title))
        for key in ("rmse", "mean", "median", "std", "min", "max", "sse"):
            if key in stats:
                self("    %-7s %12.6f %s" % (key, stats[key], unit))

    def save(self, path):
        with open(path, "w") as fh:
            fh.write("\n".join(self.lines).rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="APE of the filtered-frame SLAM trajectory against the raw-frame one.")
    ap.add_argument("filtered_name", nargs="?", default=FILTERED_NAME,
                    help="name of the filtered-frame folder, e.g. "
                         "opti_multi1_free_nuc3_recall100_K1 (default: %(default)s)")
    ap.add_argument("--trial", default=None,
                    help="override the trial name (default: parsed from the filtered name)")
    ap.add_argument("--id", default=None,
                    help="override the <id> folder (default: parsed from the filtered name)")
    ap.add_argument("--workspace", default=WORKSPACE, help="default: %(default)s")
    ap.add_argument("--traj-kind", choices=["cam", "kf"], default="cam",
                    help="compare the per-frame or the keyframe trajectory (default: %(default)s)")
    ap.add_argument("--raw-traj", default=None, help="explicit path to the raw-frame trajectory")
    ap.add_argument("--filt-traj", default=None, help="explicit path to the filtered-frame trajectory")
    ap.add_argument("--time-unit", choices=["auto", "s", "ms", "us", "ns"], default="auto",
                    help="unit of the timestamp column (default: %(default)s)")
    ap.add_argument("--max-diff", type=float, default=0.01,
                    help="association tolerance in seconds (default: %(default)s)")

    al = ap.add_mutually_exclusive_group()
    al.add_argument("--no-align", action="store_true",
                    help="compare in the trajectories' own frames, without alignment")
    al.add_argument("--align-scale", action="store_true",
                    help="Sim(3) alignment (also solves for scale) instead of SE(3)")
    ap.add_argument("--n-to-align", type=int, default=-1,
                    help="align using only the first N associated poses (default: all)")
    ap.add_argument("--plot", action="store_true", help="write a comparison plot")
    ap.add_argument("--plot-path", default=None,
                    help="where to write the plot (default: <filtered_name>_filt_vs_raw.png "
                         "in the LRFilter directory)")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                    help="figure size in inches (default: 10x10, square)")
    ap.add_argument("--ref-label", default="No filter",
                    help="legend label for the raw trajectory (default: %(default)s)")
    ap.add_argument("--est-label", default=None,
                    help="legend label for the filtered trajectory "
                         "(default: read from the filter tag, e.g. "
                         "'LRFilter (Recall=95, K=2)')")
    ap.add_argument("--report", default=None,
                    help="where to write the report (default: <filtered_name>_eval.txt "
                         "in the LRFilter directory)")
    ap.add_argument("--no-report", action="store_true",
                    help="print to stdout only, do not write the report file")
    args = ap.parse_args()

    from evo.core import metrics, sync
    from evo.core.metrics import PoseRelation

    # The name may arrive as a path or with a trailing slash; keep the basename.
    filtered_name = os.path.basename(args.filtered_name.rstrip("/")) or args.filtered_name
    parsed_trial, parsed_id = parse_trial_and_id(filtered_name)
    trial = args.trial or parsed_trial
    trial_id = args.id or parsed_id

    def slam_path(name):
        return SLAM_OUTPATH.format(workspace=args.workspace, id=trial_id,
                                   name=name, traj_kind=args.traj_kind)

    # The filtered bag is named after the whole filtered folder, so its trajectory
    # is too; the raw trajectory only ever carries the bare trial name.
    raw_path = args.raw_traj or slam_path("%s_raw" % trial)
    filt_path = args.filt_traj or slam_path("%s_filt" % filtered_name)

    ref_label = args.ref_label
    est_label = args.est_label or filter_label(filtered_name, trial)

    rep = Report()
    rep("=" * 78)
    rep("filter evaluation: %s" % filtered_name)
    rep("  trial %s | id %s | %s trajectory" % (trial, trial_id, args.traj_kind))
    rep("=" * 78)

    traj_ref, ref_unit = load_tum(raw_path, args.time_unit)
    traj_est, est_unit = load_tum(filt_path, args.time_unit)
    rep.describe("raw (ref)", traj_ref, raw_path, ref_unit)
    rep.describe("filt (est)", traj_est, filt_path, est_unit)

    n_ref, n_est = traj_ref.num_poses, traj_est.num_poses
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=args.max_diff)
    rep("")
    rep("associated %d pose pairs (max_diff %.3f s) -- %.1f%% of raw, %.1f%% of filt"
        % (traj_ref.num_poses, args.max_diff,
           100.0 * traj_ref.num_poses / n_ref, 100.0 * traj_ref.num_poses / n_est))
    if traj_ref.num_poses < 3:
        die("only %d pose pairs associated -- check --max-diff and the timestamp "
            "units of both files" % traj_ref.num_poses)

    if args.no_align:
        mode = "none (raw frames as-is)"
    else:
        traj_est.align(traj_ref, correct_scale=args.align_scale,
                       n=args.n_to_align)
        mode = "Sim(3), scale solved" if args.align_scale else "SE(3), scale fixed"
        if args.n_to_align > 0:
            mode += " (first %d poses)" % args.n_to_align
    rep("alignment: %s" % mode)

    data = (traj_ref, traj_est)

    ape_t = metrics.APE(PoseRelation.translation_part)
    ape_t.process_data(data)
    rep.stats("APE -- translation", ape_t.get_all_statistics(), "m")

    ape_r = metrics.APE(PoseRelation.rotation_angle_deg)
    ape_r.process_data(data)
    rep.stats("APE -- rotation", ape_r.get_all_statistics(), "deg")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from evo.tools import plot
        from evo.tools.settings import SETTINGS
        import matplotlib.pyplot as plt

        # Applied after importing evo.tools.plot, which styles matplotlib at
        # import time and would otherwise clobber these.
        plt.style.use("tableau-colorblind10")
        plt.rcParams.update({
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 18,
            "legend.fontsize": 20,
        })

        out = args.plot_path or os.path.join(LRFILTER_DIR,
                                             "%s_filt_vs_raw.png" % filtered_name)

        # 1.5x matplotlib's (and evo's) default line width of 1.5.
        line_width = 1.5 * 1.5
        SETTINGS.plot_linewidth = line_width

        # Colour-map in centimetres rather than metres.
        err_cm = ape_t.error * 100.0

        fig_w, fig_h = args.figsize if args.figsize else (10.0, 10.0)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = plot.prepare_axis(fig, plot.PlotMode.xy)

        plot.traj(ax, plot.PlotMode.xy, traj_ref, "--", "gray", ref_label)
        # The estimate is drawn as a LineCollection, which contributes no legend
        # handle of its own -- add an empty proxy line so it appears in the legend.
        ax.plot([], [], "-", color="#1f4e79", linewidth=line_width, label=est_label)

        before = set(fig.axes)
        plot.traj_colormap(ax, traj_est, err_cm, plot.PlotMode.xy,
                           min_map=float(err_cm.min()), max_map=float(err_cm.max()))
        # traj_colormap() builds its own colorbar and does not hand it back, so
        # find the axis it added to label it.
        cbar_ax = None
        for cax in fig.axes:
            if cax not in before:
                cax.set_ylabel("APE (cm)")
                cbar_ax = cax

        # evo passes linewidth through inconsistently; force it on what was drawn.
        for line in ax.lines:
            line.set_linewidth(line_width)
        for coll in ax.collections:
            coll.set_linewidth(line_width)

        # ax.set_title("ORB-SLAM3 Trajectory With and Without LRFilter")
        ax.legend(frameon=True, loc="best")

        # Square axes box, but with "datalim" so x and y keep the same metric
        # scale -- a trajectory drawn with unequal scales is misleading.
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_box_aspect(1)

        # NB: no tight_layout()/subplots_adjust here -- traj_colormap() attaches a
        # colorbar, which installs a layout engine that rejects both. Let that
        # engine settle, freeze it, then match the colorbar to the squared axes
        # (it is sized before set_box_aspect shrinks the plot, so it ends up
        # taller than the thing it annotates).
        fig.canvas.draw()
        if cbar_ax is not None:
            fig.set_layout_engine("none")
            box, cbox = ax.get_position(), cbar_ax.get_position()
            cbar_ax.set_position([cbox.x0, box.y0, cbox.width, box.height])
        fig.savefig(out, dpi=130)
        rep("")
        rep("plot -> %s" % out)

    if not args.no_report:
        report_path = args.report or os.path.join(LRFILTER_DIR,
                                                  "%s_eval.txt" % filtered_name)
        report_dir = os.path.dirname(os.path.abspath(report_path))
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        rep.save(report_path)
        print("")
        print("report -> %s" % report_path)
    print("")


if __name__ == "__main__":
    main()
