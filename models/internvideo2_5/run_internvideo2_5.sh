#!/bin/bash
set -e

exec python3 \
    /workspace/models/internvideo2_5/internvideo2_5_infer.py \
    "$@"
