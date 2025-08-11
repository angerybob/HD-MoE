# HD-MoE: Hybrid and Dynamic Parallelism for MoE LLMs on 3D Near-Memory Processing

This repository contains the implementation of **HD-MoE**, a hybrid and dynamic parallelism framework designed to optimize MoE (Mixture-of-Experts) LLM inference on 3D Near-Memory Processing (3D NMP) architectures. This work has been accepted by the **2025 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)**. HD-MoE achieves significant speedups over traditional parallelism strategies by balancing computation load, minimizing communication overhead, and adapting to dynamic expert activation patterns.


## Overview

Large Language Models (LLMs) with Mixture-of-Expert (MoE) architectures offer superior performance with reduced computation costs but face challenges in memory bandwidth and efficient parallelization. 3D Near-Memory Processing (3D NMP) architectures address memory-bound issues with high-bandwidth memory stacking, but their distributed nature introduces new mapping and scheduling challenges.

HD-MoE tackles these challenges through:
- **Offline Hybrid Parallel Mapping**: Combines Tensor Parallelism (TP) and Expert Parallelism (EP) to balance computation and communication.
- **Online Dynamic Scheduling**: Adapts to real-time expert activation patterns to optimize resource utilization.

Experimental results show HD-MoE achieves 1.1×–1.8× speedup over TP, 1.1×–1.5× over EP, and 1.0×–1.4× over hybrid TP-EP baselines.


## Quick Start

### 1. 环境搭建（Environment Setup）

#### 创建并激活conda环境
```bash
# 创建环境
conda create -n hdmoe python=3.10
# 激活环境
conda activate hdmoe
```

#### 克隆仓库并安装依赖
```bash
# 克隆仓库
git clone git@github.com:angerybob/HD-MoE.git
# 进入仓库目录
cd HD-MoE
# 安装依赖
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```


### 2. 生成部署策略（Generate Deployment Strategy）

通过优化脚本生成针对特定硬件和模型的部署策略，后台运行并输出日志：
```bash
nohup optimizer.sh > script.log 2>&1 &
```

- **输出位置**：
  - 部署策略结果：`results/` 文件夹
  - 每层输出日志：`logs/` 文件夹

- **参数配置**：
  可在 `optimizer.sh` 中修改以下配置以适配不同场景：
  ```bash
  # 硬件配置
  comp=10.0          # 算力（TFLOPS）
  BW=25.0            # 带宽（GB/s）
  mesh_shapeX=4      # 2D mesh X维度尺寸
  mesh_shapeY=8      # 2D mesh Y维度尺寸
  # 任务配置
  batch=128          # 批次大小
  model="qwen"       # 模型类型（如"qwen"、"mixtral"等）
  ```
  脚本中for循环的层数也要根据模型具体配置修改 

### 3. 评估部署策略（Evaluate Deployment Strategy）

使用评估脚本验证部署策略的性能，支持端到端 latency、消融实验和动态调度评估：

#### 评估命令
```bash
# 评估端到端TBT latency（对应文章中时间间隔指标）
python evaluation/scripts/e2e.py

# 消融实验（验证各模块作用）
python evaluation/scripts/ablation.py

# 动态调度策略评估
python evaluation/scripts/dynamic.py
```

- **评估结果位置**：`evaluation/results/` 文件夹
- **评估前配置**：需在对应脚本中修改硬件配置（算力、带宽等）、模型类型及数据集，确保与生成的部署策略匹配。


### 4. 结果可视化（Visualization）

通过绘图脚本将评估结果可视化，生成与论文对应的关键图表：

```bash
# 绘制不同硬件配置下的端到端加速比（对应Fig. 8）
python evaluation/draw/draw.py

# 绘制不同mesh尺寸下的性能（对应Fig. 9）
python evaluation/draw/draw_mesh.py

# 绘制节点平衡优化的加速比（对应Fig. 10）
python evaluation/draw/ablation_draw.py

# 绘制节点平衡对计算延迟的优化（对应Fig. 11）
python evaluation/draw/ablation2_draw.py

# 绘制节点级资源利用平衡（对应Fig. 12）
python evaluation/draw/balance.py

# 绘制链路平衡优化的加速比（对应Fig. 13）
python evaluation/draw/ablation3_draw.py

# 绘制链路级资源利用平衡（对应Fig. 14）
python evaluation/draw/balance2.py

# 绘制动态调度策略性能（对应Fig. 15 (a)）
python evaluation/draw/dynamic_draw.py

# 绘制不同预广播专家数量下的动态调度性能（对应Fig. 15 (b)）
python evaluation/draw/dynamic_draw2.py
```

- **图表输出位置**：`evaluation/figs/` 文件夹


## 核心模块说明（Core Modules）

- **`node_allocation.py`**：实现 `MoE3DPNMOptimizer` 类，封装了文章中提出的Node-Link Balance优化算法。

- **`simulator.py`**：主要优化流程实现，模拟3D NMP架构下的MoE推理过程，包含计算与通信开销建模。

- **`baseline.py`**：提供基线策略（TP、EP、混合TP-EP）的实现，用于快速对比优化结果。

- **`expert_trace/`**：存储不同模型（如Mixtral、DeepSeek等）的专家激活统计数据，用于部署策略的生成与优化。


## 支持的模型与数据集（Supported Models & Datasets）

- **模型**：支持MoE架构模型（如Qwen、Mixtral、DeepSeek等），可通过 `expert_trace/` 中的专家激活数据扩展新模型。
- **数据集**：默认使用MT Bench数据集（广泛用于LLM性能评估），可在评估脚本中替换为其他数据集。


## Citation

若使用本仓库代码，请引用相关工作：
```bibtex
@article{hdmoe,
  title={HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing},
  author={Haochen Huang, Shuzhang Zhong, Zhe Zhang, Shuangchen Li, Dimin Niu, Hongzhong Zheng, Runsheng Wang and Meng Li},
  booktitle={Proceedings of the 44th IEEE/ACM International Conference on Computer-Aided Design},
  pages={1--9},
  year={2025}
}
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.