#!/bin/bash

# 配置
MAX_JOBS=6               # 最大并发数
comp=10.0
BW=25.0
batch=128
mesh_shapeX=4
mesh_shapeY=8
model="qwen"
LOG_DIR="/data/home/haochenhuang/deployment/logs/reasoning_${model}_${comp}TFLOPS_${BW}GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"              # 日志目录
SCRIPT_PATH="/data/home/haochenhuang/deployment/simulator.py"
result_DIR="/data/home/haochenhuang/deployment/results/reasoning_${model}_${comp}_TFLOPS_${BW}_GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"
# 创建日志目录
mkdir -p "$LOG_DIR"
mkdir -p "$result_DIR"

# 并行执行任务
for layer_id in {0..27}; do
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 1  # 等待空闲进程
    done

    echo "启动 layer_id = $layer_id"
    nohup python3 "$SCRIPT_PATH" --layer-id "$layer_id" --comp $comp --comm $BW --batch $batch --mesh-shape "($mesh_shapeX,$mesh_shapeY)" --model $model> "${LOG_DIR}/log_layer_${layer_id}.txt" 2>&1 &
done

wait  # 等待所有任务完成
echo "所有任务执行完毕！日志保存在 $LOG_DIR/"