# HD-MoE: Hybrid and Dynamic Parallelism for MoE LLMs on 3D Near-Memory Processing  

For full technical details, please refer to our paper: 
[arXiv version](https://arxiv.org/abs/2509.09420) | 
[ICCAD (IEEE) version](https://ieeexplore.ieee.org/abstract/document/11240984)

This repository contains the implementation of **HD-MoE**, a hybrid and dynamic parallelism framework designed to optimize Mixture-of-Experts (MoE) Large Language Model (LLM) inference on 3D Near-Memory Processing (3D NMP) architectures. This work has been accepted by the **2025 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)**. HD-MoE achieves significant speedups over traditional parallelism strategies by balancing computation load, minimizing communication overhead, and adapting to dynamic expert activation patterns.  


## Overview  

Large Language Models (LLMs) with Mixture-of-Expert (MoE) architectures deliver superior performance while reducing computation costs. However, they face critical challenges related to memory bandwidth limitations and inefficient parallelization. 3D Near-Memory Processing (3D NMP) architectures address memory-bound bottlenecks through high-bandwidth memory stacking, but their distributed nature introduces new challenges in mapping and scheduling.  

HD-MoE tackles these challenges through two core components:  
- **Offline Hybrid Parallel Mapping**: Combines Tensor Parallelism (TP) and Expert Parallelism (EP) to balance computation workload and communication overhead.  
- **Online Dynamic Scheduling**: Adapts to real-time expert activation patterns to optimize resource utilization dynamically.  

Experimental results demonstrate that HD-MoE achieves a 1.1×–1.8× speedup over standalone TP, a 1.1×–1.5× speedup over standalone EP, and a 1.0×–1.4× speedup over hybrid TP-EP baselines.  


## Quick Start  

### 1. Environment Setup  

#### Create and Activate a Conda Environment  
```bash
# Create environment
conda create -n hdmoe python=3.10
# Activate environment
conda activate hdmoe
```

#### Clone the Repository and Install Dependencies  
```bash
# Clone the repository
git clone git@github.com:angerybob/HD-MoE.git
# Navigate to the repository directory
cd HD-MoE
# Install dependencies
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```


### 2. Generate Deployment Strategy  

Run the optimization script to generate a deployment strategy tailored to specific hardware and model configurations. Execute the script in the background and log outputs:  
```bash
nohup optimizer.sh > script.log 2>&1 &
```

- **Output Locations**:  
  - Deployment strategy results: `results/` folder  
  - Per-layer output logs: `logs/` folder  

- **Parameter Configuration**:  
  Modify the following settings in `optimizer.sh` to adapt to different scenarios:  
  ```bash
  # Hardware configuration
  comp=10.0          # Computational power (TFLOPS)
  BW=25.0            # Bandwidth (GB/s)
  mesh_shapeX=4      # 2D mesh size (X-dimension)
  mesh_shapeY=8      # 2D mesh size (Y-dimension)
  # Task configuration
  batch=128          # Batch size
  model="qwen"       # Model type (e.g., "qwen", "mixtral")
  ```  
  Additionally, adjust the number of layers in the `for` loop within the script to match the specific configuration of your target model.  


### 3. Evaluate Deployment Strategy  

Use the evaluation scripts to validate the performance of the generated deployment strategy. Supported evaluations include end-to-end latency, ablation studies, and dynamic scheduling assessment.  

#### Evaluation Commands  
```bash
# Evaluate end-to-end TBT latency (corresponds to the time interval metric in the paper)
python evaluation/scripts/e2e.py

# Ablation study (validate the impact of individual modules)
python evaluation/scripts/ablation.py

# Evaluate dynamic scheduling strategy
python evaluation/scripts/dynamic.py
```

- **Evaluation Result Location**: `evaluation/results/` folder  
- **Pre-Evaluation Configuration**: Before running evaluations, modify hardware settings (computational power, bandwidth, etc.), model type, and dataset in the corresponding scripts to ensure alignment with the generated deployment strategy.  


### 4. Result Visualization  

Use the plotting scripts to visualize evaluation results and generate key figures consistent with those in the paper:  

```bash
# Plot end-to-end speedups across different hardware configurations (corresponds to Fig. 8)
python evaluation/draw/draw.py

# Plot performance across different mesh sizes (corresponds to Fig. 9)
python evaluation/draw/draw_mesh.py

# Plot speedups from node balance optimization (corresponds to Fig. 10)
python evaluation/draw/ablation_draw.py

# Plot optimization of computation latency via node balance (corresponds to Fig. 11)
python evaluation/draw/ablation2_draw.py

# Plot node-level resource utilization balance (corresponds to Fig. 12)
python evaluation/draw/balance.py

# Plot speedups from link balance optimization (corresponds to Fig. 13)
python evaluation/draw/ablation3_draw.py

# Plot link-level resource utilization balance (corresponds to Fig. 14)
python evaluation/draw/balance2.py

# Plot dynamic scheduling strategy performance (corresponds to Fig. 15 (a))
python evaluation/draw/dynamic_draw.py

# Plot dynamic scheduling performance with different pre-broadcast expert counts (corresponds to Fig. 15 (b))
python evaluation/draw/dynamic_draw2.py
```

- **Figure Output Location**: `evaluation/figs/` folder  


## Core Modules  

- **`node_allocation.py`**: Implements the `MoE3DPNMOptimizer` class, which encapsulates the Node-Link Balance optimization algorithm proposed in the paper.  

- **`simulator.py`**: Implements the core optimization workflow, simulating MoE inference on 3D NMP architectures (including computation and communication overhead modeling).  

- **`baseline.py`**: Provides implementations of baseline strategies (TP, EP, hybrid TP-EP) for quick performance comparison with HD-MoE.  

- **`expert_trace/`**: Stores expert activation statistics for different models (e.g., Mixtral, DeepSeek), which are used to generate and optimize deployment strategies.  


## Supported Models & Datasets  

- **Models**: Supports MoE-architecture LLMs (e.g., Qwen, Mixtral, DeepSeek). New models can be integrated by adding their expert activation data to the `expert_trace/` directory.  
- **Datasets**: Uses the MT Bench dataset by default (a widely adopted benchmark for LLM performance evaluation). Other datasets can be substituted in the evaluation scripts.  


## Citation  

If you use the code in this repository, please cite the associated work:  
```bibtex
@INPROCEEDINGS{11240984,
  author={Huang, Haochen and Zhong, Shuzhang and Zhang, Zhe and Li, Shuangchen and Niu, Dimin and Zheng, Hongzhong and Wang, Runsheng and Li, Meng},
  booktitle={2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD)}, 
  title={HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Costs;Three-dimensional displays;Tensors;Computational modeling;Memory management;Bandwidth;Parallel processing;Dynamic scheduling;Distance measurement;Computational efficiency;Automated Deployment;Mixture-of-Experts;3D Near-Memory Processing},
  doi={10.1109/ICCAD66269.2025.11240984}}
```  


## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
