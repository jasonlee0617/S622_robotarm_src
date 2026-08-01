#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

keep='^(realsense2_gz_description|fairino_hardware|depthai-ros|realsense-ros)$'
for dir in build install log; do
  [ -d "$dir" ] || continue
  find "$dir" -mindepth 1 -maxdepth 1 -printf '%f\0' | while IFS= read -r -d '' name; do
    if [[ "$name" =~ $keep ]]; then
      echo "keep $dir/$name"
    else
      rm -rf -- "$dir/$name"
    fi
  done
done
