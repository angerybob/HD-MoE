#!/bin/bash

# 获取传入的参数
comp=$2
comm=$4
batch=$6

# 打印参数，用于调试
echo "Computation throughput (TFLOPS): $comp"
echo "Communication bandwidth (GB/s): $comm"
echo "Batch size: $batch"

python3 /data/home/haochenhuang/deployment/evaluation/test.py --comp $comp --comm $comm --batch $batch

/data/home/haochenhuang/deployment/astra-sim/build/astra_ns3/build.sh -r