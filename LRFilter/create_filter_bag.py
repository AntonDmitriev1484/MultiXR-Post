#!/usr/bin/env python
r"""
create_filter_bag.py -- rebuild a raw trial bag using a filtered set of camera frames.

Takes the name of a filtered-frame folder, e.g.

    opti_multi1_free_nuc3_recall100_K1
    \_________________/\____________/
        <trial>            filter tag

The trial name is everything up to and including the "nuc<N>" token, and <id> is
that N, which together locate the raw bag:

    <WORKSPACE>/<id>/collect/<trial>_raw/<trial>_raw.bag

Every message is copied verbatim EXCEPT the camera image topics, which are
replaced by the PNGs in the filtered folder (one per kept frame, named by its
header timestamp in seconds).  The result is written next to the raw bag as
<filtered_name>_filt.bag, so several filter settings for one trial can coexist.

Needs ROS1 (Ubuntu 20.04 / noetic).  Run it inside the u20_tools container:

    ./create_filter_bag.py opti_multi1_free_nuc3_recall100_K1 --docker

or, by hand:

    docker run --rm --entrypoint /bin/bash \
        -v /home/antond2/Desktop/Research:/research u20_tools:latest \
        -c 'source /opt/ros/noetic/setup.bash && python3 /research/MultiXR-Post/LRFilter/create_filter_bag.py'
"""

from __future__ import print_function

import argparse
import csv
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------- defaults --

FILTERED_NAME = "opti_multi1_free_nuc3_recall100_K1"

FILTERED_DATA_DIR = "/home/antond2/Desktop/Research/SLAMKF-Eval/{filtered_name}"
RAW_DATA_PATH = ("/home/antond2/Desktop/Research/MultiXR-Post/{id}/collect/"
                 "{trial}_raw/{trial}_raw.bag")

LEFT_TOPIC = "/camera/camera/infra1/image_rect_raw"
RIGHT_TOPIC = "/camera/camera/infra2/image_rect_raw"

# The filtered frames carry infra1's header stamps verbatim, so an exact hit is
# expected; the tolerance only absorbs float-repr rounding in the filenames.
MATCH_TOL_S = 1e-3

DOCKER_IMAGE = "u20_tools:latest"
DOCKER_MOUNT_HOST = "/home/antond2/Desktop/Research"
DOCKER_MOUNT_GUEST = "/research"


# ------------------------------------------------------------------ helpers --

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def die(msg):
    sys.stderr.write("error: " + msg + "\n")
    sys.exit(1)


def run_in_docker(argv):
    """Re-exec this script inside the ROS1 container with the research tree mounted."""
    here = os.path.abspath(__file__)
    if not here.startswith(DOCKER_MOUNT_HOST + os.sep):
        die("--docker expects this script to live under %s (it is at %s)"
            % (DOCKER_MOUNT_HOST, here))
    guest_script = DOCKER_MOUNT_GUEST + here[len(DOCKER_MOUNT_HOST):]

    inner = ("source /opt/ros/noetic/setup.bash && python3 %s %s"
             % (guest_script, " ".join(_shquote(a) for a in argv)))
    # --user keeps the output bag owned by the caller instead of root.
    cmd = ["docker", "run", "--rm",
           "--user", "%d:%d" % (os.getuid(), os.getgid()),
           "--entrypoint", "/bin/bash",
           "-v", "%s:%s" % (DOCKER_MOUNT_HOST, DOCKER_MOUNT_GUEST),
           DOCKER_IMAGE, "-c", inner]
    log("+ " + " ".join(_shquote(c) for c in cmd))
    return subprocess.call(cmd)


def parse_trial_and_id(filtered_name):
    """opti_multi1_free_nuc3_recall100_K1 -> ('opti_multi1_free_nuc3', '3').

    The trial name is the prefix up to and including the 'nuc<N>' token; whatever
    the filter appended after it (recall100_K1, ...) is the filter tag.
    """
    m = re.search(r"nuc(\d+)", filtered_name)
    if not m:
        die("could not parse a trial name out of %r -- expected a 'nuc<N>' token, "
            "e.g. opti_multi1_free_nuc3_recall100_K1 (use --raw-bag/--filtered-dir "
            "to bypass the naming convention)" % filtered_name)
    return filtered_name[:m.end()], m.group(1)


def _shquote(s):
    if s and all(c.isalnum() or c in "-_./:=" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def host_to_guest(path):
    """Rewrite a host path into its in-container equivalent when running in docker."""
    if os.path.exists(path):
        return path
    if path.startswith(DOCKER_MOUNT_HOST + os.sep):
        alt = DOCKER_MOUNT_GUEST + path[len(DOCKER_MOUNT_HOST):]
        if os.path.exists(alt):
            return alt
    return path


def load_filtered_frames(filtered_dir):
    """Return [(timestamp_seconds, png_path)] sorted by timestamp.

    Prefers manifest.csv (full float precision) and falls back to parsing the
    filenames, which are the same stamps at repr() precision.
    """
    manifest = os.path.join(filtered_dir, "manifest.csv")
    stamps = {}
    if os.path.isfile(manifest):
        with open(manifest) as fh:
            for row in csv.DictReader(fh):
                stamps[row["filename"]] = float(row["timestamp"])
        log("read %d entries from manifest.csv" % len(stamps))

    frames = []
    for name in sorted(os.listdir(filtered_dir)):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(filtered_dir, name)
        if name in stamps:
            frames.append((stamps[name], path))
        else:
            try:
                frames.append((float(os.path.splitext(name)[0]), path))
            except ValueError:
                log("  skipping %s (filename is not a timestamp and it is not "
                    "in manifest.csv)" % name)
    frames.sort(key=lambda f: f[0])
    return frames


def png_to_image_msg(path, template, cv2, np):
    """Load a PNG and stuff it into a sensor_msgs/Image cloned from `template`."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        die("could not read image %s" % path)

    enc = template.encoding
    if enc == "mono8":
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)
        channels = 1
    elif enc in ("bgr8", "rgb8"):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)
        channels = 3
    elif enc == "mono16":
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint16)
        channels = 1
    else:
        die("unsupported source encoding %r on %s -- extend png_to_image_msg()"
            % (enc, template.header.frame_id))

    img = np.ascontiguousarray(img)
    msg = type(template)()
    msg.header = template.header          # seq / stamp / frame_id preserved
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = enc
    msg.is_bigendian = template.is_bigendian
    msg.step = img.shape[1] * channels * img.dtype.itemsize
    msg.data = img.tobytes()
    return msg


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild a raw bag with filtered camera frames.")
    ap.add_argument("filtered_name", nargs="?", default=FILTERED_NAME,
                    help="name of the filtered-frame folder, e.g. "
                         "opti_multi1_free_nuc3_recall100_K1 (default: %(default)s)")
    ap.add_argument("--id", default=None,
                    help="override the <id> collect folder (default: parsed from the name)")
    ap.add_argument("--trial", default=None,
                    help="override the trial name (default: parsed from the filtered name)")
    ap.add_argument("--raw-bag", default=None, help="path to <trial>_raw.bag")
    ap.add_argument("--filtered-dir", default=None, help="directory of filtered PNG frames")
    ap.add_argument("--out", default=None,
                    help="output bag (default: <filtered_name>_filt.bag beside the raw bag)")
    ap.add_argument("--left-topic", default=LEFT_TOPIC,
                    help="image topic the filtered frames came from (default: %(default)s)")
    ap.add_argument("--right-topic", default=RIGHT_TOPIC,
                    help="the other stereo image topic (default: %(default)s)")
    ap.add_argument("--keep-right-all", action="store_true",
                    help="keep every right-camera frame instead of dropping the ones "
                         "whose stamp was filtered out of the left camera")
    ap.add_argument("--tol", type=float, default=MATCH_TOL_S,
                    help="stamp match tolerance in seconds (default: %(default)s)")
    ap.add_argument("--compression", choices=["none", "bz2", "lz4"], default="none",
                    help="output bag compression (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing output bag")
    ap.add_argument("--docker", action="store_true",
                    help="re-run this script inside the %s ROS1 container" % DOCKER_IMAGE)
    args = ap.parse_args()

    if args.docker:
        passthrough = [a for a in sys.argv[1:] if a != "--docker"]
        sys.exit(run_in_docker(passthrough))

    try:
        import rosbag
        import numpy as np
        import cv2
    except ImportError as exc:
        die("%s -- this needs ROS1 noetic; re-run with --docker, or:\n"
            "  docker run --rm --user $(id -u):$(id -g) --entrypoint /bin/bash \\\n"
            "    -v %s:%s %s \\\n"
            "    -c 'source /opt/ros/noetic/setup.bash && python3 %s%s'"
            % (exc, DOCKER_MOUNT_HOST, DOCKER_MOUNT_GUEST, DOCKER_IMAGE,
               DOCKER_MOUNT_GUEST, "/MultiXR-Post/LRFilter/create_filter_bag.py"))

    filtered_name = args.filtered_name.rstrip("/")
    filtered_name = os.path.basename(filtered_name) or filtered_name
    parsed_trial, parsed_id = parse_trial_and_id(filtered_name)
    trial = args.trial or parsed_trial
    trial_id = args.id or parsed_id

    raw_bag = host_to_guest(
        args.raw_bag or RAW_DATA_PATH.format(id=trial_id, trial=trial))
    filtered_dir = host_to_guest(
        args.filtered_dir or FILTERED_DATA_DIR.format(filtered_name=filtered_name))
    out_bag = args.out or os.path.join(os.path.dirname(raw_bag),
                                       "%s_filt.bag" % filtered_name)
    out_bag = host_to_guest(os.path.dirname(out_bag)) + "/" + os.path.basename(out_bag)

    if not os.path.isfile(raw_bag):
        die("raw bag not found: %s" % raw_bag)
    if not os.path.isdir(filtered_dir):
        die("filtered frame dir not found: %s" % filtered_dir)
    if os.path.exists(out_bag) and not args.overwrite:
        die("output already exists (pass --overwrite): %s" % out_bag)

    log("filtered name: %s  ->  trial %s, id %s" % (filtered_name, trial, trial_id))
    log("raw bag      : %s" % raw_bag)
    log("filtered dir : %s" % filtered_dir)
    log("output bag   : %s" % out_bag)
    log("")

    # -- 1. the filtered frames -------------------------------------------------
    frames = load_filtered_frames(filtered_dir)
    if not frames:
        die("no PNG frames in %s" % filtered_dir)
    log("filtered frames: %d  (%.6f .. %.6f)"
        % (len(frames), frames[0][0], frames[-1][0]))

    # -- 2. index the raw bag's left-camera stamps ------------------------------
    log("indexing %s ..." % args.left_topic)
    bag_in = rosbag.Bag(raw_bag, "r")
    left_stamps = []          # header stamp, nanoseconds
    stamp_to_bagtime = {}
    template = None
    for _, msg, t in bag_in.read_messages(topics=[args.left_topic]):
        ns = msg.header.stamp.to_nsec()
        left_stamps.append(ns)
        stamp_to_bagtime[ns] = t
        if template is None:
            template = msg
    if template is None:
        bag_in.close()
        die("no messages on %s in the raw bag" % args.left_topic)
    left_sorted = sorted(left_stamps)
    log("  %d left-camera messages in the raw bag" % len(left_stamps))

    # -- 3. filtered frame -> raw left-camera message ---------------------------
    import bisect
    tol_ns = int(round(args.tol * 1e9))
    keep = {}                 # header-stamp ns -> png path
    unmatched = []
    for ts, path in frames:
        want = int(round(ts * 1e9))
        i = bisect.bisect_left(left_sorted, want)
        best, best_err = None, None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(left_sorted):
                err = abs(left_sorted[j] - want)
                if best_err is None or err < best_err:
                    best, best_err = left_sorted[j], err
        if best is None or best_err > tol_ns:
            unmatched.append((ts, path))
        else:
            keep[best] = path

    log("matched %d/%d filtered frames to raw left-camera messages "
        "(max stamp error %s)"
        % (len(keep), len(frames), "n/a" if not keep else "within %.1f us" % (tol_ns / 1e3)))
    if unmatched:
        log("  !! %d filtered frames have no raw counterpart within %.4fs; "
            "they will be inserted as new messages:" % (len(unmatched), args.tol))
        for ts, path in unmatched[:10]:
            log("     %.7f  %s" % (ts, os.path.basename(path)))
        if len(unmatched) > 10:
            log("     ... and %d more" % (len(unmatched) - 10))

    # Bag-receipt time for synthesised messages: reuse the raw topic's median
    # header->receipt latency so they land in the right spot in the stream.
    lat = sorted(stamp_to_bagtime[s].to_nsec() - s for s in left_stamps)
    median_lat = lat[len(lat) // 2]

    pending = []              # (bagtime ns, topic, msg) for synthesised frames
    if unmatched:
        from genpy import Time as _RosTime
        for ts, path in unmatched:
            ns = int(round(ts * 1e9))
            tmpl = type(template)()
            tmpl.header = type(template.header)()
            tmpl.header.seq = template.header.seq
            tmpl.header.frame_id = template.header.frame_id
            tmpl.header.stamp = _RosTime(ns // 10**9, ns % 10**9)
            tmpl.encoding = template.encoding
            tmpl.is_bigendian = template.is_bigendian
            pending.append((ns + median_lat, args.left_topic,
                            png_to_image_msg(path, tmpl, cv2, np)))
        pending.sort(key=lambda p: p[0])

    # -- 4. stream the raw bag into the filtered bag ----------------------------
    total = bag_in.get_message_count()
    counts = {"copied": 0, "left_kept": 0, "left_dropped": 0,
              "right_kept": 0, "right_dropped": 0, "inserted": 0}
    image_topics = set([args.left_topic, args.right_topic])

    log("")
    log("writing %s ..." % out_bag)
    tmp_out = out_bag + ".part"
    from genpy import Time as RosTime
    pi = 0
    try:
        with rosbag.Bag(tmp_out, "w", compression=args.compression) as bag_out:
            for n, (topic, msg, t) in enumerate(bag_in.read_messages(), 1):
                # flush any synthesised frames that belong before this message
                while pi < len(pending) and pending[pi][0] <= t.to_nsec():
                    bt, tp, m = pending[pi]
                    bag_out.write(tp, m, RosTime(bt // 10**9, bt % 10**9))
                    counts["inserted"] += 1
                    pi += 1

                if topic == args.left_topic:
                    ns = msg.header.stamp.to_nsec()
                    if ns in keep:
                        bag_out.write(topic, png_to_image_msg(keep[ns], msg, cv2, np), t)
                        counts["left_kept"] += 1
                    else:
                        counts["left_dropped"] += 1
                elif topic == args.right_topic and not args.keep_right_all:
                    if msg.header.stamp.to_nsec() in keep:
                        bag_out.write(topic, msg, t)
                        counts["right_kept"] += 1
                    else:
                        counts["right_dropped"] += 1
                else:
                    bag_out.write(topic, msg, t)
                    if topic in image_topics:
                        counts["right_kept"] += 1
                    else:
                        counts["copied"] += 1

                if n % 2000 == 0:
                    log("  %d/%d messages" % (n, total))

            while pi < len(pending):
                bt, tp, m = pending[pi]
                bag_out.write(tp, m, RosTime(bt // 10**9, bt % 10**9))
                counts["inserted"] += 1
                pi += 1
    finally:
        bag_in.close()

    os.rename(tmp_out, out_bag)

    log("")
    log("done -> %s (%.1f MB)" % (out_bag, os.path.getsize(out_bag) / 1e6))
    log("  non-image messages copied : %d" % counts["copied"])
    log("  left  frames kept/dropped : %d / %d" % (counts["left_kept"], counts["left_dropped"]))
    log("  right frames kept/dropped : %d / %d%s"
        % (counts["right_kept"], counts["right_dropped"],
           "  (--keep-right-all)" if args.keep_right_all else ""))
    if counts["inserted"]:
        log("  frames inserted (no raw match): %d" % counts["inserted"])


if __name__ == "__main__":
    main()
