#!/usr/bin/env bash
set -euo pipefail

COLLECTIVEAPI_PATH="${HOME}/collectiveapi"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
export PYTHONPATH="/home/tuyen/astra-sim/extern/graph_frontend:${PYTHONPATH:-}"

python3 "$COLLECTIVEAPI_PATH/msccl-tools/examples/mscclang/alltoall_allpairs.py" 8 1 \
  > "$SCRIPT_DIR/alltoall_8ranks.xml"

mkdir -p "$SCRIPT_DIR/alltoall_8ranks"

python3 "$COLLECTIVEAPI_PATH/chakra_converter/et_converter.py" \
  --input_filename "$SCRIPT_DIR/alltoall_8ranks.xml" \
  --output_filename "$SCRIPT_DIR/alltoall_8ranks/alltoall_8ranks"