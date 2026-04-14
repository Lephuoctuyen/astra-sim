#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
TARGET_WORKLOAD="DLRM_HybridParallel"

# Run visualizer
(
mkdir -p ${SCRIPT_DIR}/workload
chakra_converter Text \
    --input="${SCRIPT_DIR:?}"/text_workloads/"${TARGET_WORKLOAD:?}.txt" \
    --output="${SCRIPT_DIR:?}"/workload/"${TARGET_WORKLOAD:?}" \
    --num-npus=128 \
    --num-passes=1
)
