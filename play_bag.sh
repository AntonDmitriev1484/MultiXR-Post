#!/bin/bash

# Require exactly one argument
if [ $# -lt 2 ]; then
  echo "Usage: $0 <folder> <trial>"
  exit 1
fi

DIR="$1"
A="$2"

rosbag play "/data/${DIR}/collect/${A}/${A}.bag"