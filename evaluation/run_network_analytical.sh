#!/bin/bash
set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples/network_analytical"

# paths
ASTRA_SIM="/data/home/haochenhuang/deployment/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="/data/home/haochenhuang/deployment/evaluation/workload"
SYSTEM="/data/home/haochenhuang/deployment/evaluation/system_cfg.json"
NETWORK="/data/home/haochenhuang/deployment/evaluation/sample_8nodes_1D.json"
REMOTE_MEMORY="/data/home/haochenhuang/deployment/evaluation/remote_memory_cfg.json"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Analytical Network Backend..."
echo ""

# Compile
/data/home/haochenhuang/deployment/astra-sim/build/astra_ns3/build.sh

echo ""
echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim Example with Analytical Network Backend..."
echo ""

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}"

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."
