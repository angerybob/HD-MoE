#!/bin/bash

# GPU资源配置
TOTAL_GPUS=8
GPUS_PER_MODEL=8
MAX_CONCURRENT=$((TOTAL_GPUS / GPUS_PER_MODEL))  # 计算结果为1

# 日志配置
LOG_DIR="/data/home/haochenhuang/deployment/trace/logs"
TRACE_DIR="/data/home/haochenhuang/deployment/evaluation"
mkdir -p "$LOG_DIR"
cd /data/home/haochenhuang/FastChat/fastchat/llm_judge

declare -a TASKS=(
    # DeepSeek-V2-Lite-Chat 任务


    "--model-path /opt/pretrained_models/DeepSeek-V2-Lite-Chat --model-id 10002 \
     --trace ${TRACE_DIR}/experts_extraction_deepseek.json \
     --question-begin 51 --question-end 56"

    "--model-path /opt/pretrained_models/DeepSeek-V2-Lite-Chat --model-id 10003 \
     --trace ${TRACE_DIR}/experts_stem_deepseek.json \
     --question-begin 61 --question-end 66"

    "--model-path /opt/pretrained_models/DeepSeek-V2-Lite-Chat --model-id 10004 \
     --trace ${TRACE_DIR}/experts_humanities_deepseek.json \
     --question-begin 71 --question-end 76"

  
)

# 任务执行控制
current_jobs=0
for task in "${TASKS[@]}"; do
    while [ "$current_jobs" -ge "$MAX_CONCURRENT" ]; do
        echo "等待GPU资源释放... (当前运行任务: $current_jobs)"
        sleep 60
        current_jobs=$(pgrep -f gen_model_answer.py | wc -l)
    done

    # 提取任务参数用于日志命名
    model_id=$(echo "$task" | grep -oP -- '--model-id \d+' | cut -d' ' -f2)
    q_begin=$(echo "$task" | grep -oP -- '--question-begin \d+' | cut -d' ' -f2)
    q_end=$(echo "$task" | grep -oP -- '--question-end \d+' | cut -d' ' -f2)
    
    # 启动任务
    echo "启动任务: model-id=$model_id (问题 $q_begin-$q_end)"
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u /data/home/haochenhuang/FastChat/fastchat/llm_judge/gen_model_answer.py $task \
        > "${LOG_DIR}/model_${model_id}_q${q_begin}-${q_end}.log" 2>&1 &

    ((current_jobs++))
    sleep 10  # 避免任务启动冲突
done

wait
echo "所有任务完成！日志保存在 $LOG_DIR/ trace文件保存在 $TRACE_DIR/"