#!/bin/bash

# 定义参数值
COMP=30
COMM=20
BATCH=128
# 打印参数，用于调试
echo "Computation throughput (TFLOPS): $COMP"
echo "Communication bandwidth (GB/s): $COMM"
echo "Batch size: $BATCH"
# 运行 TP.sh 脚本，并传递参数
bash /data/home/haochenhuang/deployment/evaluation/TP.sh --comp $COMP --comm $COMM --batch $BATCH

# 运行 EP.sh 脚本，并传递参数
bash /data/home/haochenhuang/deployment/evaluation/EP.sh --comp $COMP --comm $COMM --batch $BATCH

# 运行 ours.sh 脚本，并传递参数
bash /data/home/haochenhuang/deployment/evaluation/ours.sh --comp $COMP --comm $COMM --batch $BATCH