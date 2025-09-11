#!/bin/bash

# Configuration
MAX_JOBS=6               # Maximum number of concurrent processes
comp=10.0                # Computational power (TFLOPS)
BW=25.0                  # Bandwidth (GB/s)
batch=128                # Batch size
mesh_shapeX=4            # 2D mesh dimension X
mesh_shapeY=8            # 2D mesh dimension Y
model="qwen"             # Model type
LOG_DIR="logs/reasoning_${model}_${comp}TFLOPS_${BW}GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"              # 日志目录
SCRIPT_PATH="simulator.py"
result_DIR="results/reasoning_${model}_${comp}_TFLOPS_${BW}_GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"

# Create log and result directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$result_DIR"

# Execute tasks in parallel
for layer_id in {0..27}; do
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 1  # Wait for idle processes
    done

    echo "Starting layer_id = $layer_id"
    nohup python3 "$SCRIPT_PATH" --layer-id "$layer_id" --comp $comp --comm $BW --batch $batch --mesh-shape "($mesh_shapeX,$mesh_shapeY)" --model $model> "${LOG_DIR}/log_layer_${layer_id}.txt" 2>&1 &
done

wait  # Wait for all background tasks to complete
echo "All tasks completed! Logs are saved in $LOG_DIR/"