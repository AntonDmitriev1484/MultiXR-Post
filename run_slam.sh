#!/usr/bin/env bash

# Require exactly one argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 <folder>"
  exit 1
fi

DIR="$1"

roslaunch "/data/${DIR}/orbslam/launch/stereoi_launch.launch" dir:="${DIR}"