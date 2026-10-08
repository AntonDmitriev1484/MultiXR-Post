#!/usr/bin/env bash
# Copy 2/ 3/ 4/ to DEST, skipping every directory directly under <n>/collect/
# whose name doesn't contain "opti_multi1". Everything else (orbslam/, post/,
# synth_failures/, loose files in collect/) is copied as-is.
#
# Usage: ./copy_send_parham.sh [--dry-run]
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST=/home/antond2/Desktop/SendParham1

mkdir -p "$DEST"
cd "$SRC_ROOT"

# Patterns are anchored at the transfer root, which for source "2" is the
# parent dir, so "/*/collect/..." matches 2/collect/..., 3/collect/..., etc.
# Trailing "/" restricts a pattern to directories. First match wins.
rsync -a --info=progress2 "$@" \
    --include='/*/collect/*opti_multi1*/' \
    --exclude='/*/collect/*/' \
    2 3 4 "$DEST/"
