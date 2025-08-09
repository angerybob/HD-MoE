
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import seaborn as sns

batch=int(128*4)
model="ds"
if model=="ds":
    E,e,SE,h,IS,mlp_first,num_layers=64,6,0,2048,1408,True,26  #DeepSeekMoE 
elif model=="mixtral":
    E,e,SE,h,IS,mlp_first,num_layers=8,2,0,4096,14336,False,32 #Mixtral
elif model=="qwen":
    E,e,SE,h,IS,mlp_first,num_layers=64,8,0,3584,18944,False,28 #Qwen2

dataset=["reasoning","math","coding","writing","roleplay"][0]
sample1=["reasoning","math","coding","writing","roleplay"][4]
mesh_shape=(4,8)
D=mesh_shape[0]*mesh_shape[1]
data_path=f'expert_trace/ds/predict/experts_{dataset}_{model}_pre.json'
#data_path="/data/home/haochenhuang/deployment/evaluation/experts_reasoning_mixtral.json"
sample_path=f'expert_trace/ds/predict/experts_{sample1}_{model}_pre.json'
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        data,pre=data1["selected_experts"],data1["predict_experts"]

    with open(sample_path, 'r', encoding='utf-8') as f1:
        sample2 = json.load(f1)
        sample,pre_sample=sample2["selected_experts"],sample2["predict_experts"]
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
try:
    with open('expert_trace/qwen/experts_reasoning_qwen.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
    
num_expert=E

num_count=np.zeros((num_layers, num_expert))

# 遍历每一层
for layer_id in range(num_layers):
    #遍历每一个专家激活组
    for sub_list in data1[str(layer_id+mlp_first)]:
        # 遍历专家激活组中的每个专家
        for num in sub_list:
            num_count[layer_id][num]+=1

#----------------------------------------------------------------------------------


reuse=np.zeros((num_layers, len(data["1"])-1))
for layer_id in range(num_layers):
    #遍历每一个专家激活组
    for sub_id in range(len(data[str(layer_id+mlp_first)])):
        if sub_id>0:
            reuse_p=len(set(data[str(layer_id+mlp_first)][sub_id])&set(data[str(layer_id+mlp_first)][sub_id-1]))/len(data[str(layer_id+mlp_first)][sub_id])
            reuse[layer_id,sub_id-1]=reuse_p




#----------------------------------------------------------------------------------

group_count = {}
layer_id=20

group_count=np.zeros((num_expert, num_expert))



# 统计专家组共激活频率
for ref_expert in range(num_expert):
    for sub_list in data1[str(layer_id+mlp_first)]:
        # 遍历专家激活组中的每个专家
        if ref_expert in sub_list:
            for num in sub_list:
                group_count[ref_expert][num] += 1/((num_count[layer_id][ref_expert]))

            
group_count[group_count>=0.95]=0
    
#----------------------------------------------------------------------------------
predict=np.zeros((num_layers, len(data["1"])))
for layer_id in range(num_layers):
    #遍历每一个专家激活组
    for sub_id in range(len(data[str(layer_id+mlp_first)])):
        reuse_p=len(set(data[str(layer_id+mlp_first)][sub_id])&set(pre[str(layer_id+mlp_first)][sub_id]))/len(data[str(layer_id+mlp_first)][sub_id])
        #pdb.set_trace()
        predict[layer_id,sub_id]=reuse_p
    
    
    
plt.rcParams.update({
    "font.size": 16,
     "axes.labelweight": "normal",
     "axes.labelsize": 20,
    "legend.frameon": True,
    "lines.linewidth": 3
})

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.5, hspace=2.5) 
# 调整刻度线方向
for ax in axs.flat:
    ax.tick_params(direction='in')  # 所有刻度线朝内

# 第一张图：neuron_cum, mixtral_cum, deepseek_cum
heatmap = sns.heatmap(
    num_count[:,32:]/len(data1["1"]), 
    annot=False,
    fmt=".2f",
    cmap="inferno",
    linewidths=0.5,
    cbar_kws={'label': 'Frequency'},
    ax=axs[0,0]  # 关键：指定子图位置
)

# 设置颜色条字体
cbar = heatmap.collections[0].colorbar

cbar.set_label('Frequency', 
               fontsize=20,
               labelpad=10,
               x=-5,# 调整标签与颜色条的间距
               y=1.1,         # 垂直位置（0=底部，1=顶部）
               rotation=0,  # 旋转标签（垂直显示）
               ha="right",
               va='top'
              )
cbar.ax.tick_params(labelsize=24)

# 设置坐标轴标签
axs[0,0].set_xlabel("Expert Index", fontsize=24)
axs[0,0].set_ylabel("Layer Index", fontsize=24)



# X轴刻度（每隔6个显示）
step_x = 6
selected_ticks_x = range(0, num_expert//2, step_x)
axs[0,0].set_xticks(selected_ticks_x)
axs[0,0].set_xticklabels(selected_ticks_x, rotation=0, fontsize=20)

# Y轴刻度（每隔4层显示）
step_y = 4
selected_ticks_y = range(0, num_layers, step_y)
axs[0,0].set_yticks(selected_ticks_y)
axs[0,0].set_yticklabels(selected_ticks_y, rotation=0, fontsize=20)

# 添加子图标签 (a)
axs[0,0].text(
    0.5, -0.32, '(a)', 
    ha='center', va='center', 
    transform=axs[0,0].transAxes, 
    fontsize=20
)



# 第二张图： 
axs[0, 1].plot(range(num_layers), np.mean(reuse,axis=1), label='Reuse Probability', color='darkblue', linestyle='-' , marker='o',  markersize=8)
axs[0, 1].set_xlabel('Layer Index', fontsize=24)
axs[0, 1].set_ylabel('Reuse Probability', fontsize=24)
axs[0, 1].set_ylim(0, 1)
axs[0, 1].text(0.5, -0.32, '(b)', ha='center', va='center', transform=axs[0, 1].transAxes, fontsize=20)
axs[0, 1].tick_params(axis='both', which='major', labelsize=24)
heatmap = sns.heatmap(
    group_count[:32,:32], 
    annot=False,
    fmt=".2f",
    cmap="inferno",
    linewidths=0.5,
    cbar_kws={'label': 'Affinity'},
    ax=axs[1,0]  # 关键：指定子图位置
)

# 设置颜色条字体
cbar = heatmap.collections[0].colorbar
cbar.set_label('Affinity', fontsize=28)
cbar.ax.tick_params(labelsize=24)
cbar.set_label('Frequency', 
               fontsize=20,
               labelpad=10,
               x=5,# 调整标签与颜色条的间距
               y=1.1,         # 垂直位置（0=底部，1=顶部）
               rotation=0,  # 旋转标签（垂直显示）
               ha="right",
               va='top'
              )
# 设置坐标轴标签
axs[1,0].set_xlabel("Expert Index", fontsize=24)
axs[1,0].set_ylabel("Expert Index", fontsize=24)



# X轴刻度（每隔6个显示）
step_x = 6
selected_ticks_x = range(0, num_expert//2, step_x)
axs[1,0].set_xticks(selected_ticks_x)
axs[1,0].set_xticklabels(selected_ticks_x, rotation=0, fontsize=20)

# Y轴刻度（每隔4层显示）
step_y = 6
selected_ticks_y = range(0, num_expert//2, step_y)
axs[1,0].set_yticks(selected_ticks_y)
axs[1,0].set_yticklabels(selected_ticks_y, rotation=0, fontsize=20)

# 添加子图标签 (a)
axs[1,0].text(
    0.5, -0.32, '(c)', 
    ha='center', va='center', 
    transform=axs[1,0].transAxes, 
    fontsize=20
)

axs[1, 1].plot(range(num_layers), np.mean(predict,axis=1), label='Accurancy of Prediction', color='darkgreen', linestyle='-' , marker='o',  markersize=8)
axs[1, 1].set_xlabel('Layer Index', fontsize=24)
axs[1, 1].set_ylabel('Accurancy of Prediction', fontsize=24)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].text(0.5, -0.32, '(d)', ha='center', va='center', transform=axs[1, 1].transAxes, fontsize=20)
axs[1, 1].tick_params(axis='both', which='major', labelsize=24)

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
save_dir = 'evaluation/figs/motivation/motivation.png'
plt.savefig(save_dir)
print(f"Figure saved at {save_dir}")