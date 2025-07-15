import numpy as np
from scipy.optimize import linprog
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import json
import ast
import pdb
import math
from tqdm import tqdm
import random
from heapq import heappush, heappop
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real
from scipy.optimize import linear_sum_assignment

class MoE3DPNMOptimizer:
    def __init__(self, routing_trace, E=64, e=6, SE=2,h=2048,IS=1408, B=128, D=64, BW=25e9, comp=10e12, num_layers=26, mlp_first=True):
        # 参数初始化
        self.E = E+SE          # 专家数量
        self.e = e+SE
        self.h = h          # 隐藏层维度
        self.IS=IS
        self.B = B          # Batch size
        self.D = D          # 3D PNM设备维度
        self.BW = BW        # 带宽 (Bytes/s)
        self.comp = comp    # 计算能力 (FLOP/s)
        self.layer = num_layers # number of moe layers
        self.SE=SE
        self.mlp_first=mlp_first
        self.routing_trace=routing_trace
        self.R_cc = (self.BW * self.IS * self.e) / (2 * self.D * self.comp)
        # 专家激活频率
        self.f = np.zeros((self.layer, self.E))
        self.f[:][self.E-self.SE:]=1
        #pdb.set_trace()  
        for layer_id in range(self.layer):
            for sub_list in routing_trace[str([layer_id,layer_id+1][self.mlp_first])]:
                # 遍历子列表中的每个数字
                for num in sub_list:
                    self.f[layer_id][num]+=1
        self.f=self.f/len(routing_trace[str(1)])
        self.fg = self._generate_co_activation(routing_trace)
        #self.fusion_map=self.fusion_detection()
        self.route_cache = {}
        self.X=8
        self.Y=8
        self.M=np.zeros((self.D,self.X,self.Y))
        #self.aggregators = self._find_optimal_aggregator()
        #pdb.set_trace()
        
    def _find_optimal_aggregator(self):
        """为简化实现，默认选择中心节点作为聚合点"""
        # 实际实现应基于专家分布，这里简化为几何中心
        x_center = np.mean([d[0] for d in self.M])
        y_center = np.mean([d[1] for d in self.M])
        min_dist = float('inf')
        best_d = 0
        for d, (x, y) in enumerate(self.M):
            dist = abs(x - x_center) + abs(y - y_center)
            if dist < min_dist:
                min_dist = dist
                best_d = d
        return best_d
    
    def _generate_co_activation(self,routing_trace):
        """生成专家共激活频率矩阵"""
        fg = {}
        fg_pruning={}
        k=self.e
        print("Begin to sampling activation data...")
        '''for k in tqdm(range(2, self.e+1)):
            #print(k)
            if k==2 or k==self.e:'''
        fg[k] = {}
        fg_pruning[k] = {}
        #print((1.5*(k//2)+0.5*math.ceil(k/2))//2)
        #threshold = len(list(combinations(range(self.E), int((1.5*(k//2)+0.5*math.ceil(k/2))//2)))) / len(list(combinations(range(self.E), k)))
        #pdb.set_trace()
        #threshold = (1.5*len(list(combinations(range(self.E), int(k//2))))+0.5*len(list(combinations(range(self.E), math.ceil(k/2)))))/2 / len(list(combinations(range(self.E), k)))
        #threshold = (len(list(combinations(range(self.E), k//2))) / len(list(combinations(range(self.E), k))))*len(routing_trace["1"])
        for layer_id in tqdm(range(self.layer)):
            #print(layer_id)
            fg[k][layer_id]={}
            fg_pruning[k][layer_id]={}
            for sub_list in routing_trace[str([layer_id,layer_id+1][self.mlp_first])]:
                temp_list=list(sorted(list(sub_list)))
                temp_list.extend(list(range(self.E-self.SE,self.E)))
                               
                list_key=str(temp_list)
                #pdb.set_trace()
                if list_key in fg[k][layer_id]:
                    fg[k][layer_id][list_key] += 1
                else:
                    fg[k][layer_id][list_key] = 1
            # 计算所有 value 的和
            #total_sum = sum(fg[k][layer_id].values())
            # pruning
            #if layer_id>6:
                #pdb.set_trace()

            # 归一化
            for key in fg[k][layer_id]:
                fg[k][layer_id][key] /= len(routing_trace[str([layer_id,layer_id+1][self.mlp_first])])
                '''if k==self.e:
                    fg_pruning[k][layer_id][key] = fg[k][layer_id][key] 
                elif fg[k][layer_id][key]>threshold:
                    fg_pruning[k][layer_id][key] = fg[k][layer_id][key]'''        
            #pdb.set_trace()
        return fg

    # ----------------- 性能分析模型 -----------------
    def compute_time(self, P):
        """计算时间 t_comp (公式1)"""
        compute_load = np.sum(P * self.f[:,:, None] * self.B * 2 * self.h*self.IS, axis=1)
        '''single_comm = np.zeros((self.layer,self.D))
        for layer_id, layer_fg in self.fg[self.e].items():
            for list_key, freq in layer_fg.items():
                group = ast.literal_eval(list_key)
                devices = np.sum(P[layer_id][group]>0,axis=0)
                #redundant = freq * self.B * self.h * np.maximum((devices-1),0)
                redundant = 2*freq * self.B * self.h *self.IS* (devices>0)
                #single_comm[layer_id] -= redundant
                single_comm[layer_id] += redundant/np.sum(devices>0)
                #pdb.set_trace()
        #pdb.set_trace()'''
        return np.max(compute_load / self.comp, axis=1)
    
    def compute_time_dynamic(self, P,comp_map):
        compute_load = np.sum(P * comp_map[None,:,None], axis=1)
        #pdb.set_trace()
        return np.max(compute_load / self.comp, axis=1)

    def comm_time(self, P):
        """通信时间 t_comm (公式2)"""
        # 单专家通信量
        #single_comm = np.sum((P > 0) * self.f[:,:, None] * self.B * self.h, axis=1)
        #pdb.set_trace()
        single_comm = np.zeros((self.layer,self.D))
        # 扣除共激活冗余量
        '''
        for k, k_fg in self.fg.items():
            for layer_id, layer_fg in k_fg.items():
                for list_key, freq in layer_fg.items():
                    group = ast.literal_eval(list_key)
                    devices = np.any(P[layer_id][group, :], axis=0)
                    redundant = freq * self.B * self.h * devices
                    single_comm[layer_id] -= redundant
        '''

        
        for layer_id, layer_fg in self.fg[self.e].items():
            for list_key, freq in layer_fg.items():
                group = ast.literal_eval(list_key)
                devices = np.sum(P[layer_id][group]>0,axis=0)
                #redundant = freq * self.B * self.h * np.maximum((devices-1),0)
                redundant = freq * self.B * self.h * (devices>0)
                #single_comm[layer_id] -= redundant
                single_comm[layer_id] += redundant
        #for l,layer_group in self.routing_trace.items():
            #for group in layer_group
        #pdb.set_trace()
        return 4*np.max(single_comm, axis=1) / (self.BW)
    
    def comm_time_dynamic(self, P,random_samples):

        single_comm = np.zeros((self.layer,self.D))
    
        for l in range(self.layer):
            for d in range(self.D):
                expert_gropus=list(np.nonzero(P[l, :, d])[0])
                for sublist in random_samples:
                    if bool(set(sublist).intersection(set(expert_gropus))):
                        single_comm[l,d]+=self.h
        


        return 4*np.max(single_comm, axis=1) / (self.BW)
    
    def _get_xy_path(self, src, dst):
        """带缓存的XY路由路径生成"""
        src=tuple((int(src[0]),int(src[1])))
        dst=tuple((int(dst[0]),int(dst[1])))
        #src = tuple(map(int, src)) if isinstance(src, np.ndarray) else tuple(src)
        #dst = tuple(map(int, dst)) if isinstance(dst, np.ndarray) else tuple(dst) 
        #pdb.set_trace()
        cache_key = (tuple(src), tuple(dst))
        if cache_key not in self.route_cache:
            path = []
            current = src
            while current[0] != dst[0]:
                next_node = (current[0] + (1 if dst[0] > current[0] else -1), current[1])
                path.append((current, next_node))
                current = next_node
            while current[1] != dst[1]:
                next_node = (current[0], current[1] + (1 if dst[1] > current[1] else -1))
                path.append((current, next_node))
                current = next_node
            self.route_cache[cache_key] = path
        return self.route_cache[cache_key]
    
    def _simulate_comm(self,M, layer_data,layer_id,P,chunks=20):
        """基于离散事件仿真的通信时间计算"""
        # 初始化数据结构
        link_schedule = defaultdict(list)  # {link: [(start_time, end_time)]}
        event_queue = []
        for sublist in self.fg[self.e][layer_id].keys():
            sublist=ast.literal_eval(sublist)
            non_zero_coords = []
            for i in sublist:
                # 找到当前二维矩阵中不为 0 的元素的坐标
                #pdb.set_trace()
                d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])
                #pdb.set_trace()
                
                for x, y in zip(rows, cols):
                    if (x, y) not in non_zero_coords:
                        non_zero_coords.append((x, y))

            d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            #idex=np.random.randint(0,len(x_s))
            idex=0
            x_center,y_center=x_s[idex],y_s[idex]
            #pdb.set_trace()
            aggregator=tuple((x_center,y_center))
            data_size = self.fg[self.e][layer_id][str(sublist)]*self.B*self.h
            

            for p in non_zero_coords:
                path = self._get_xy_path(p, aggregator)
                for _ in range(chunks):
                    heappush(event_queue, (0.0, data_size/chunks, path))
                    #pdb.set_trace()
        '''
        # 生成所有传输任务
        for device in range(self.layer):
            expert_gropus=list(np.nonzero(P[layer_id, :, device])[0])
            #print(f"generating tasks for {device}")
            for sublist in self.fg[self.e][layer_id].keys():
                sublist=ast.literal_eval(sublist)
                if bool(set(sublist).intersection(set(expert_gropus))):
                    non_zero_coords = []
                    for i in sublist:
                        # 找到当前二维矩阵中不为 0 的元素的坐标
                        #pdb.set_trace()
                        d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])
                        
                        for x, y in zip(rows, cols):
                            if (x, y) not in non_zero_coords:
                                non_zero_coords.append((x, y))
                    x_center = int(np.mean([d[0] for d in non_zero_coords]))
                    y_center = int(np.mean([d[1] for d in non_zero_coords]))
                    
                    aggregator=tuple((x_center,y_center))
                    data_size = 4*self.fg[self.e][layer_id][str(sublist)]*self.B*self.h
                    path = self._get_xy_path(tuple(np.nonzero(M[device])), aggregator)
                    for _ in range(chunks):
                        heappush(event_queue, (0.0, data_size/chunks, path))
        '''
        # 处理事件队列
        max_finish_time = 0
        count=0
        while event_queue:
            current_time, remaining_data, remaining_path = heappop(event_queue)
            
            if not remaining_path:
                
                max_finish_time = max(max_finish_time, current_time)
                #print(max_finish_time)
                continue
                
            current_link = remaining_path[0]
            available_bw = self.BW
            
            # 查找当前链路可用时间窗口
            last_end = 0
            for start, end in sorted(link_schedule[current_link]):
                if last_end <= current_time < start:
                    available_window = start - current_time
                    break
                last_end = end
                current_time=max(current_time,end)
            else:
                available_window = float('inf')
                
            # 计算本次传输量
            trans_time = remaining_data / available_bw
            actual_trans = min(trans_time, available_window)
            
            # 记录链路占用
            new_start = current_time
            new_end = current_time + actual_trans
            link_schedule[current_link].append((new_start, new_end))
            
            # 更新剩余数据和事件
            if actual_trans < trans_time:
                new_remaining = remaining_data - actual_trans * available_bw
                heappush(event_queue, (new_end, new_remaining, remaining_path))
            else:
                heappush(event_queue, (new_end, remaining_data, remaining_path[1:]))
        link_load = defaultdict(float)
        for link, time_windows in link_schedule.items():
            for start, end in time_windows:
                link_load[link] += end - start  
        return max_finish_time,link_load
    
    def comm_time_acc(self, M,P,layer_id,chunks=1):
        """基于离散事件仿真的通信时间计算"""
        
        # 计算各设备发送数据量
        layer_data = np.zeros((self.D))


        
        #print("processing the expert freq...")
        for list_key, freq in self.fg[self.e][layer_id].items():
            group = ast.literal_eval(list_key)
            devices = np.sum(P[layer_id][group]>0,axis=0)
            #redundant = freq * self.B * self.h * np.maximum((devices-1),0)
            redundant = freq * self.B * self.h * (devices>0)
            #single_comm[layer_id] -= redundant
            layer_data += redundant
        #print("finish processing!")
        # 执行离散事件仿真
        comm_time,link = self._simulate_comm(M,layer_data,layer_id,P,chunks)
        #layer_comm_times.append(comm_time)
            
        return comm_time,link 
    
    
    def _simulate_comm_dynamic(self,M, layer_data,layer_id,P,random_samples,chunks=20):
        """基于离散事件仿真的通信时间计算"""
        # 初始化数据结构
        link_schedule = defaultdict(list)  # {link: [(start_time, end_time)]}
        event_queue = []
        #print(link_schedule)
        # 生成所有传输任务
        '''
        for device in np.nonzero(layer_data)[0]:
            expert_gropus=list(np.nonzero(P[layer_id, :, device])[0])
            for sublist in random_samples:
                #sublist=ast.literal_eval(sublist)
                #print(event_queue)
                if bool(set(sublist).intersection(set(expert_gropus))):
                    #print("111")
                    non_zero_coords = []
                    for i in sublist:
                        # 找到当前二维矩阵中不为 0 的元素的坐标
                        #pdb.set_trace()
                        d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])
                        
                        for x, y in zip(rows, cols):
                            if (x, y) not in non_zero_coords:
                                non_zero_coords.append((x, y))
                    x_center = int(np.mean([d[0] for d in non_zero_coords]))
                    y_center = int(np.mean([d[1] for d in non_zero_coords]))
                    d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][max(sublist)])])
                    x_center,y_center=x_s[0],y_s[0]
                    #pdb.set_trace()
                    aggregator=tuple((x_center,y_center))
                    data_size = 4*self.h
                    path = self._get_xy_path(tuple(np.nonzero(M[device])), aggregator)
                    for _ in range(chunks):
                        heappush(event_queue, (0.0, data_size/chunks, path))
                    #print(event_queue)
        '''


        for sublist in random_samples:

            non_zero_coords = []
            for i in sublist:
                # 找到当前二维矩阵中不为 0 的元素的坐标
                #pdb.set_trace()
                d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])
                
                for x, y in zip(rows, cols):
                    if (x, y) not in non_zero_coords:
                        non_zero_coords.append((x, y))

            d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            #idex=np.random.randint(0,len(x_s))
            while len(x_s)==0:
                d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            idex=0
            x_center,y_center=x_s[idex],y_s[idex]
            #pdb.set_trace()
            aggregator=tuple((x_center,y_center))
            data_size = self.h
            

            for p in non_zero_coords:
                path = self._get_xy_path(p, aggregator)
                for _ in range(chunks):
                    heappush(event_queue, (0.0, data_size/chunks, path))

            '''
            start=(self.D-1,self.D-1)
            for p in non_zero_coords:
                if p[0]+p[1]<start[0]+start[1]:
                    start=p
            non_zero_coords.pop(non_zero_coords.index(start))
            src=start
            while non_zero_coords:
                dis=np.inf
                for p in non_zero_coords:
                    if abs(src[0]-p[0])+abs(src[1]-p[1])<dis:
                        dst=p
                        dis=abs(src[0]-p[0])+abs(src[1]-p[1])
                path = self._get_xy_path(src, dst)
                non_zero_coords.pop(non_zero_coords.index(dst))
                
                for _ in range(chunks):
                    heappush(event_queue, (0.0, data_size/chunks, path))
                src=dst
                '''
                #print(event_queue)
        # 处理事件队列
        max_finish_time = 0
        while event_queue:
            #print(len(event_queue))
            current_time, remaining_data, remaining_path = heappop(event_queue)
            
            if not remaining_path:
                #print(f"{len(event_queue)} remaining...")
                max_finish_time = max(max_finish_time, current_time)
                continue
                
            current_link = remaining_path[0]
            available_bw = self.BW
            
            # 查找当前链路可用时间窗口
            last_end = 0
            for start, end in sorted(link_schedule[current_link]):
                if last_end <= current_time < start:
                    available_window = start - current_time
                    break
                last_end = end
                current_time=max(current_time,end)
            else:
                available_window = float('inf')
                
            # 计算本次传输量
            trans_time = remaining_data / available_bw
            actual_trans = min(trans_time, available_window)
            
            # 记录链路占用
            new_start = current_time
            new_end = current_time + actual_trans
            link_schedule[current_link].append((new_start, new_end))
            
            # 更新剩余数据和事件
            if actual_trans < trans_time:
                new_remaining = remaining_data - actual_trans * available_bw
                heappush(event_queue, (new_end, new_remaining, remaining_path))
            else:
                heappush(event_queue, (new_end, remaining_data, remaining_path[1:]))
                
        return max_finish_time
    
    def comm_time_acc_dynamic(self, M,P,layer_id,random_samples,chunks=1):
        """基于离散事件仿真的通信时间计算"""
        
        # 计算各设备发送数据量
        layer_data = np.zeros((self.D))


        

        for list_key, freq in self.fg[self.e][layer_id].items():
            group = ast.literal_eval(list_key)
            devices = np.sum(P[layer_id][group]>0,axis=0)
            #redundant = freq * self.B * self.h * np.maximum((devices-1),0)
            redundant = freq * self.B * self.h * (devices>0)
            #single_comm[layer_id] -= redundant
            #pdb.set_trace()
            layer_data += redundant
            
        # 执行离散事件仿真
        comm_time = self._simulate_comm_dynamic(M,layer_data,layer_id,P,random_samples,chunks)
        #layer_comm_times.append(comm_time)
            
        return comm_time
    
    # ----------------- 最优放置策略 -----------------
    def expert_affinity(self, i, j,layer_id):
        """专家亲和性评估 (公式5)"""
        f_ij = self.fg[2][layer_id].get(str(list(sorted([i,j]))), 0)
        return f_ij / np.sqrt(self.f[layer_id][i] * self.f[layer_id][j])
    
    def fusion_detection(self):
        fusion_map={}
        fusion_group={}
        fused_expert={}
        for l in range(self.layer):
            fusion_group[l]={}
            fused_expert[l]=set()
            for i in range(self.E):
                for j in range(i,self.E):
                    if self.expert_affinity(i,j,l)>0.5:
                        fused_expert[l].add(i)
                        fused_expert[l].add(j)
                        if i in fusion_group[l]:
                            fusion_group[l][i].append(j)
                        else:
                            fusion_group[l][i]=[i,j]
        for l in range(self.layer):
            fusion_map[l]=[]
            for i in range(self.E): 
                if i in fused_expert[l] and i in fusion_group[l]:
                    fusion_map[l].append((fusion_group[l][i],len(fusion_group[l][i])))
                elif i not in fused_expert[l]:
                    fusion_map[l].append(([i],1))

        return fusion_map

    def prune_search_space(self):
        """搜索空间剪枝 (公式6-8)"""
        # 计算-通信比剪枝
        self.R_cc = (2*self.BW * self.IS * self.e) / (5 * self.D * self.comp)
        print(self.R_cc)
        if self.R_cc > 20:
            return "TP"  # 直接采用张量并行
        else:
            return "EP"
        # 高频共激活组剪枝
        
        valid_groups = [[[] for _ in range(self.layer)] for _ in range(2,self.e+1)]
        for k in range(2,self.e+1):
            for l in range(self.layer):
                threshold = len(list(combinations(range(self.E), k//2))) / len(list(combinations(range(self.E), k)))
                valid_groups[k-2][l].append([g for g, f in self.fg[self.e][l].items() if f > threshold])
        pdb.set_trace()
        return valid_groups

    def ilp_solver(self):
        """整数线性规划求解器（简化版）"""
        # 目标函数: min(t_comp + t_comm)
        c = np.concatenate([self.f * 4 * self.h**2 * self.B / self.comp, 
                           self.f * self.B * self.h / (self.BW * self.D)])
        
        # 约束条件: 每个专家仅部署在一个设备
        A_eq = np.kron(np.eye(self.D), np.ones((1, self.E)))
        b_eq = np.ones(self.D)
        
        # 求解
        res = linprog(c, A_eq=A_eq, B_eq=b_eq, bounds=(0,1))
        return res.x[:self.E*self.D].reshape(self.E, self.D)
    
    def EP_deployment(self,L, E, D):
        """
        生成专家部署策略矩阵 P，维度为 E×D。
        P[e][d] = a 表示第 e 个专家在第 d 个设备上部署了 a 的权重（0 ≤ a ≤ 1）。
        """
        P = np.zeros((L,E, D))
        
        if D >= E:
            # D >= E 时，每个专家分配到多个设备，设备权重均匀分布
            k, r = divmod(D, E)  # 每个专家至少分到 k 个设备，前 r 个专家多分 1 个设备
            devices = np.arange(D)  # 设备索引
            np.random.shuffle(devices)  # 随机打乱设备顺序
            start = 0
            for e in range(E):
                num_devices = k + 1 if e < r else k  # 当前专家分到的设备数
                end = start + num_devices
                assigned_devices = devices[start:end]  # 随机分配到设备
                P[:,e, assigned_devices] = 1.0 / num_devices  # 权重均匀分布
                start = end  # 更新下一个起始位置
        else:
            # D < E 时，专家尽可能均衡分配到设备，每个专家只在一个设备
            m, r = divmod(E, D)  # 每个设备至少分到 m 个专家，前 r 个设备多分 1 个专家
            experts = np.arange(E)  # 专家索引
            np.random.shuffle(experts)  # 随机打乱专家顺序
            expert_idx = 0
            for d in range(D):
                num_experts = m + 1 if d < r else m  # 当前设备分到的专家数
                assigned_experts = experts[expert_idx : expert_idx + num_experts]  # 随机分配到专家
                P[:,assigned_experts, d] = 1.0  # 权重为 1
                expert_idx += num_experts
        return P
    
    def ilp_solver_gurobi(self, l, gamma=4,time_limit=60):
        """基于Gurobi的整数线性规划求解器"""
        try:
            # 1. 创建模型
            model = gp.Model("MoE_Expert_Placement")
            
            # 2. 定义决策变量
            #P = model.addVars(self.layer, self.E, self.D, vtype=GRB.BINARY, name="P") # 二进制决策变量
            P = model.addVars(self.E, self.D, lb=0, ub=1, name="P") # 连续变量
            #pdb.set_trace()
            Z = model.addVars(self.E, self.D, vtype=GRB.BINARY, name="Z")  # 二进制变量，表示 P[l,i,c] > 0
            # 引入辅助变量 Y

            P_init=self.EP_deployment(self.layer,self.E,self.D)
            Z_init=P_init>0

            for i in range(self.E):
                for d in range(self.D):
                    P[i,d].Start=P_init[l,i,d]
                    Z[i,d].Start=Z_init[l,i,d]

            # 添加约束条件

            # 3. 定义辅助变量（用于处理max运算）
            t_comp = model.addVar(name="t_comp")
            t_comm = model.addVar(name="t_comm")

            
            # 4. 设置目标函数：最小化计算与通信时间的加权和
            model.setObjective(t_comp + 2*t_comm, GRB.MINIMIZE)
            
            # 5. 添加约束条件
            # 每个专家必须部署且仅部署在一个设备
            print("Begin to add constraint for expert placement...")
            
              
            for i in range(self.E):
                model.addConstr(gp.quicksum(P[i,c] for c in range(self.D)) == 1, 
                            f"expert_{i}_placement_in_layer_{l}")
            comp_per_expert=2 * self.h*self.IS * self.B
            max_comp = (1 / self.R_cc + 1) * (self.e) / self.D
            print("Begin to add constraint for computation node-balance...")

            for c in range(self.D):
                comp_load = gp.quicksum(P[i,c] * self.f[l,i] for i in range(self.E))
                model.addConstr(comp_load <= max_comp, f"comp_load_layer_{l}_node_{c}")
                model.addConstr(comp_load >= 0, f"min_comp_load_layer_{l}_node_{c}")
        # 计算时间约束：t_comp >= 各节点的计算时间
                model.addConstr(t_comp >= comp_load*comp_per_expert/self.comp, f"comp_time_layer_{l}_node_{c}")
            
            # 通信时间约束：t_comm >= 各节点的通信时间
            ''' single_comm = np.sum((P > 0) * self.f[:,:, None] * self.B * self.h, axis=1)
        
        # 扣除共激活冗余量
        for k, k_fg in self.fg.items():
            for layer_id, layer_fg in k_fg.items():
                for list_key, freq in layer_fg.items():
                    group = ast.literal_eval(list_key)
                    devices = np.any(P[layer_id][group, :], axis=0)
                    redundant = freq * self.B * self.h * devices
                    single_comm[layer_id] -= redundant
        
        return np.max(single_comm) / (self.BW * self.D)'''       
            print("Begin to add constraint for communication node-balance...")
            comm_per_token=self.B * self.h
            

            for c in range(self.D):
                for i in range(self.E):
                    model.addConstr(P[i,c] <= Z[i,c], f"P_leq_Z_{l}_{i}_{c}")
                # 单专家通信量
                #single_comm = gp.quicksum(Z[i,c] * self.f[layer_id,i] for i in range(self.E))
                single_comm = 0
                # 扣除共激活冗余量
                '''for k, k_fg in self.fg.items():
                    
                    for list_key, freq in k_fg[l].items():
                        group = ast.literal_eval(list_key)
                        Y = model.addVar(vtype=GRB.BINARY, name="Y")
                        for i in group:
                            model.addConstr(Y <= Z[l,i,c], f"Y_leq_Z_{l}_{i}_{c}")  # 如果 Y=1，则所有 Z[l,i,c]=1
                        model.addConstr(Y >= gp.quicksum(Z[l,i,c] for i in group) - (len(group) - 1), f"Y_geq_sum_Z_{l}_{i}_{c}")  # 如果所有 Z[l,i,c]=1，则 Y=1
                        if all(i < self.E for i in group):  # 过滤无效组
                            #group_active = gp.quicksum(Z[l,i,c] for i in group)
                            redundant = freq * self.B * self.h * Y
                            single_comm -= redundant'''
            
                for list_key, freq in self.fg[self.e][l].items():
                    Y = model.addVar(vtype=GRB.BINARY, name="Y_"+list_key+f"_placed_on_{c}_in_layer_{l}") 
                    group = ast.literal_eval(list_key)
                    # 使用 Gurobi 的 quicksum 计算 devices
                    #devices = gp.max_(gp.quicksum(Z[g,c] for g in group)-1, 0)  # 假设 P 是 3D 变量，P[layer_id][g][d] 表示 g 是否在设备 d 上
                    # 计算 redundant 并更新 single_comm
                    devices = gp.quicksum(Z[g,c] for g in group)
                    model.addConstr(Y >= devices/len(group), "expert_groups_"+list_key+f"_placed_on_{c}_in_layer_{l}")
                    redundant = freq * Y
                    single_comm += redundant
                comm_time = 4*gamma*single_comm*comm_per_token / (self.BW)
                #print(comm_time)
                model.addConstr(t_comm >= comm_time, f"comm_time_node_{c}_in_layer_{l}")
                #pdb.set_trace()
            
            # 6. 参数配置
            model.Params.TimeLimit = time_limit  # 时间限制(秒)
            model.Params.MIPGap = 0.05           # 允许5%的优化间隙
            model.Params.Threads = 8             # 使用多线程
            model.Params.Heuristics = 0.1
            #model.Params.Progress = 1
            #model.setParam("Progress", 1)
            # 7. 优化求解
            #pdb.set_trace()
            model.optimize()
            #pdb.set_trace()
            # 8. 提取结果
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.zeros((self.layer,self.E, self.D))

                for i in range(self.E):
                    for c in range(self.D):
                        solution[l,i,c] = P[i,c].X
                return solution
            else:
                print(f"No solution found. Status: {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None

        
        
    def ilp_solver_gurobi_comp(self, l, moe_model="ds",gamma=4,time_limit=60):
        """基于Gurobi的整数线性规划求解器"""
        try:
            # 1. 创建模型
            model = gp.Model("MoE_Expert_Placement_comp")
            
            # 2. 定义决策变量
            if moe_model=="mixtral":
                D=2
            else:
                D=8
            Z = model.addVars(self.E, D, vtype=GRB.BINARY, name="Z")  # 二进制变量，表示 P[l,i,c] > 0
            #pdb.set_trace()


            # 3. 定义辅助变量（用于处理max运算）
            t_comp = model.addVar(name="t_comp")


            
            # 4. 设置目标函数：最小化计算与通信时间的加权和
            model.setObjective(t_comp, GRB.MINIMIZE)
            
            # 5. 添加约束条件

            
              
            for i in range(self.E):
                model.addConstr(gp.quicksum(Z[i,c] for c in range(D)) == 1, 
                            f"expert_{i}_placement_in_layer_{l}")
            comp_per_expert=2 * self.h*self.IS * self.B



            for c in range(D):
                comp_load = gp.quicksum(Z[i,c] * self.f[l,i] for i in range(self.E))

                model.addConstr(comp_load >= 0, f"min_comp_load_layer_{l}_node_{c}")
        # 计算时间约束：t_comp >= 各节点的计算时间
                model.addConstr(t_comp >= comp_load*comp_per_expert/self.comp, f"comp_time_layer_{l}_node_{c}")
            
 
            # 6. 参数配置
            model.Params.TimeLimit = time_limit  # 时间限制(秒)
            model.Params.MIPGap = 0.05           # 允许5%的优化间隙
            model.Params.Threads = 8             # 使用多线程
            model.Params.Heuristics = 0.1
            #model.Params.Progress = 1
            #model.setParam("Progress", 1)
            # 7. 优化求解
            #pdb.set_trace()
            model.optimize()
            #pdb.set_trace()
            # 8. 提取结果
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.zeros((self.layer,self.E, D))

                for i in range(self.E):
                    for c in range(D):
                        solution[l,i,c] = Z[i,c].X
                return solution
            else:
                print(f"No solution found. Status: {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None
        
        
    def ilp_solver_gurobi_layers(self, layer_id, time_limit=60):
        """基于Gurobi的整数线性规划求解器"""
        try:
            # 1. 创建模型
            model = gp.Model("MoE_Expert_Placement")
            
            # 2. 定义决策变量
            #P = model.addVars(self.layer, self.E, self.D, vtype=GRB.BINARY, name="P") # 二进制决策变量
            P = model.addVars(self.layer,self.E, self.D, lb=0, ub=1, name="P") # 连续变量
            #pdb.set_trace()
            Z = model.addVars(self.layer,self.E, self.D, vtype=GRB.BINARY, name="Z")  # 二进制变量，表示 P[l,i,c] > 0
            # 引入辅助变量 Y
            if self.R_cc<0.1:
                P_init=self.EP_deployment(self.layer,self.E,self.D)
                Z_init=P_init>0
                for l in range(self.layer):
                    for i in range(self.E):
                        for d in range(self.D):
                            P[i,d].Start=P_init[l,i,d]
                            Z[i,d].Start=Z_init[l,i,d]

            # 添加约束条件

            # 3. 定义辅助变量（用于处理max运算）
            t_comp = model.addVars(self.layer,name="t_comp")
            t_comm = model.addVars(self.layer,name="t_comm")
            comp_sum=gp.quicksum(t_comp[l] for l in range(self.layer))
            comm_sum=gp.quicksum(t_comm[l] for l in range(self.layer))
            
            # 4. 设置目标函数：最小化计算与通信时间的加权和
            model.setObjective(comp_sum + 2*comm_sum, GRB.MINIMIZE)
            
            # 5. 添加约束条件
            # 每个专家必须部署且仅部署在一个设备
            print("Begin to add constraint for expert placement...")
            
            for l in tqdm(range(self.layer)):    
                for i in range(self.E):
                    model.addConstr(gp.quicksum(P[l,i,c] for c in range(self.D)) == 1, 
                                f"expert_{i}_placement_in_layer_{l}")
            comp_per_expert=8 * self.h**2 * self.B
            max_comp = (1 / self.R_cc + 1) * (self.e) / self.D
            print("Begin to add constraint for computation node-balance...")
            for l in tqdm(range(self.layer)):
                for c in range(self.D):
                    comp_load = gp.quicksum(P[l,i,c] * self.f[l,i] for i in range(self.E))
                    model.addConstr(comp_load <= max_comp, f"comp_load_layer_{l}_node_{c}")
                    model.addConstr(comp_load >= 0, f"min_comp_load_layer_{l}_node_{c}")
            # 计算时间约束：t_comp >= 各节点的计算时间
                    model.addConstr(t_comp[l] >= comp_load*comp_per_expert/self.comp, f"comp_time_layer_{l}_node_{c}")
            
            # 通信时间约束：t_comm >= 各节点的通信时间
            ''' single_comm = np.sum((P > 0) * self.f[:,:, None] * self.B * self.h, axis=1)
        
        # 扣除共激活冗余量
        for k, k_fg in self.fg.items():
            for layer_id, layer_fg in k_fg.items():
                for list_key, freq in layer_fg.items():
                    group = ast.literal_eval(list_key)
                    devices = np.any(P[layer_id][group, :], axis=0)
                    redundant = freq * self.B * self.h * devices
                    single_comm[layer_id] -= redundant
        
        return np.max(single_comm) / (self.BW * self.D)'''       
            print("Begin to add constraint for communication node-balance...")
            comm_per_token=self.B * self.h
            
            for l in tqdm(range(self.layer)):
                for c in range(self.D):
                    for i in range(self.E):
                        model.addConstr(P[l,i,c] <= Z[l,i,c], f"P_leq_Z_{l}_{i}_{c}")
                    # 单专家通信量
                    #single_comm = gp.quicksum(Z[i,c] * self.f[layer_id,i] for i in range(self.E))
                    single_comm = 0
                    # 扣除共激活冗余量
                    '''for k, k_fg in self.fg.items():
                        
                        for list_key, freq in k_fg[l].items():
                            group = ast.literal_eval(list_key)
                            Y = model.addVar(vtype=GRB.BINARY, name="Y")
                            for i in group:
                                model.addConstr(Y <= Z[l,i,c], f"Y_leq_Z_{l}_{i}_{c}")  # 如果 Y=1，则所有 Z[l,i,c]=1
                            model.addConstr(Y >= gp.quicksum(Z[l,i,c] for i in group) - (len(group) - 1), f"Y_geq_sum_Z_{l}_{i}_{c}")  # 如果所有 Z[l,i,c]=1，则 Y=1
                            if all(i < self.E for i in group):  # 过滤无效组
                                #group_active = gp.quicksum(Z[l,i,c] for i in group)
                                redundant = freq * self.B * self.h * Y
                                single_comm -= redundant'''
                
                    for list_key, freq in self.fg[self.e][l].items():
                        Y = model.addVar(vtype=GRB.BINARY, name="Y_"+list_key+f"_placed_on_{c}_in_layer_{l}") 
                        group = ast.literal_eval(list_key)
                        # 使用 Gurobi 的 quicksum 计算 devices
                        #devices = gp.max_(gp.quicksum(Z[g,c] for g in group)-1, 0)  # 假设 P 是 3D 变量，P[layer_id][g][d] 表示 g 是否在设备 d 上
                        # 计算 redundant 并更新 single_comm
                        devices = gp.quicksum(Z[l,g,c] for g in group)
                        model.addConstr(Y >= devices/len(group), "expert_groups_"+list_key+f"_placed_on_{c}_in_layer_{l}")
                        redundant = freq * Y
                        single_comm += redundant
                    comm_time = 5*single_comm*comm_per_token / (self.BW)
                    #print(comm_time)
                    model.addConstr(t_comm[l] >= comm_time, f"comm_time_node_{c}_in_layer_{l}")
                #pdb.set_trace()
            
            # 6. 参数配置
            model.Params.TimeLimit = time_limit  # 时间限制(秒)
            model.Params.MIPGap = 0.05           # 允许5%的优化间隙
            model.Params.Threads = 8             # 使用多线程
            model.Params.Heuristics = 0.1
            #model.Params.Progress = 1
            #model.setParam("Progress", 1)
            # 7. 优化求解
            #pdb.set_trace()
            model.optimize()
            #pdb.set_trace()
            # 8. 提取结果
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.zeros((self.layer,self.E, self.D))
                for l in range(self.layer):
                    for i in range(self.E):
                        for c in range(self.D):
                            solution[l,i,c] = P[l,i,c].X
                return solution
            else:
                print(f"No solution found. Status: {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None

    def optimize_placement(self, P, mesh_shape,time_limit=120):
        """
        优化设备在2D mesh上的放置
        :param P: E x D的专家-设备分配矩阵（二进制）
        :param mesh_shape: (X, Y)网格尺寸
        :return: D x X x Y的放置矩阵
        """
        E, D = P.shape
        X, Y = mesh_shape
        
        # 预处理专家到设备的映射
        device_for_expert = [np.where(P[e] == 1)[0].tolist() for e in range(E)]
        
        # 构建通信权重矩阵C
        C = np.zeros((D, D))

        for layer_id in self.fg[2]:
            for group_str in self.fg[self.e][layer_id]:
                group = ast.literal_eval(group_str)
                freq = self.fg[self.e][layer_id][group_str]
                # 遍历组内所有专家对
                
                e1, e2 = group[0], group[1]
                d1_group, d2_group = device_for_expert[e1], device_for_expert[e2]
                for d1 in d1_group:
                    for d2 in d2_group:
                        C[d1, d2] += freq
                        C[d2, d1] += freq  # 对称处理

        # 创建优化模型
        model = gp.Model("MoE_Device_Placement")
        
        # 决策变量：M[d, x, y]表示设备d放置在(x,y)
        M = model.addVars(D, X, Y, vtype=GRB.BINARY, name="M")
        
        # 约束1：每个设备必须放在唯一位置
        for d in range(D):
            model.addConstr(
                gp.quicksum(M[d, x, y] for x in range(X) for y in range(Y)) == 1,
                name=f"device_{d}_placement"
            )
            
        # 约束2：不同设备不能重叠
        for x in range(X):
            for y in range(Y):
                model.addConstr(
                    gp.quicksum(M[d, x, y] for d in range(D)) == 1,
                    name=f"position_{x}_{y}_unique"
                )
        
        # 定义设备的坐标变量
        x_coords = [gp.quicksum(x * M[d, x, y] for x in range(X) for y in range(Y)) 
                   for d in range(D)]
        y_coords = [gp.quicksum(y * M[d, x, y] for x in range(X) for y in range(Y)) 
                   for d in range(D)]
        
        # 创建曼哈顿距离变量
        dist_x = model.addVars(D, D, lb=0, name="dist_x")
        dist_y = model.addVars(D, D, lb=0, name="dist_y")
        
        # 添加绝对值约束
        for d1 in range(D):
            for d2 in range(D):
                model.addConstr(dist_x[d1, d2] >= x_coords[d1] - x_coords[d2], 
                              name=f"dx_{d1}_{d2}_1")
                model.addConstr(dist_x[d1, d2] >= x_coords[d2] - x_coords[d1], 
                              name=f"dx_{d1}_{d2}_2")
                model.addConstr(dist_y[d1, d2] >= y_coords[d1] - y_coords[d2], 
                              name=f"dy_{d1}_{d2}_1")
                model.addConstr(dist_y[d1, d2] >= y_coords[d2] - y_coords[d1], 
                              name=f"dy_{d1}_{d2}_2")
        
        # 构建目标函数
        objective = gp.quicksum(
            C[d1, d2] * (dist_x[d1, d2] + dist_y[d1, d2]) 
            for d1 in range(D) for d2 in range(D)
        )
        model.setObjective(objective, GRB.MINIMIZE)
        model.Params.TimeLimit = time_limit
        # 求解模型
        model.optimize()
        
        # 提取结果
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            placement = np.zeros((D, X, Y), dtype=int)
            for d in range(D):
                for x in range(X):
                    for y in range(Y):
                        if M[d, x, y].X > 0.5:
                            placement[d, x, y] = 1
            return placement
        else:
            print("Optimization failed.")
            return None

    def optimize_placement_with_gurobi(self, P,mesh_size,layer_id,time_limit):
        model = gp.Model("DevicePlacement_MST")
        D = self.D
        X, Y = mesh_size
        fg = self.fg[self.e][layer_id]
        max_M = (X-1) + (Y-1)  # 曼哈顿距离最大值

        # 设备放置变量 (D x X x Y)
        placement = model.addVars(D, X, Y, vtype=GRB.BINARY, name="place")

        # 约束: 每个设备必须放置在一个位置
        for d in range(D):
            model.addConstr(
                gp.quicksum(placement[d,x,y] for x in range(X) for y in range(Y)) == 1,
                f"device_{d}_placement"
            )
            
        # 约束2：不同设备不能重叠
        for x in range(X):
            for y in range(Y):
                model.addConstr(
                    gp.quicksum(placement[d, x, y] for d in range(D)) == 1,
                    name=f"position_{x}_{y}_unique"
                )
        # 预处理专家组的设备集合 (P已知)
        group_devices = {}
        for group_str, freq in fg.items():
            group = ast.literal_eval(group_str)
            devices = list(set([d for e in group for d in np.where(P[e] == 1)[0]])) # 获取设备ID
            group_devices[group_str] = devices

        total_cost = 0  # 总目标函数

        # 对每个专家组构建MST约束
        for group_str, devices in group_devices.items():
            if len(devices) < 2:
                continue
            
            freq = fg[group_str]
            pairs = list(combinations(devices, 2))

            # 边选择变量 (是否在生成树中)
            z = model.addVars(pairs, vtype=GRB.BINARY, name=f"z_{group_str}")

            # 生成树约束: 边数 = 节点数 - 1
            model.addConstr(z.sum() == len(devices)-1, f"mst_edge_{group_str}")

            # 单商品流约束 (确保连通性)
            root = devices[0]
            flow = model.addVars([(i,j) for i in devices for j in devices if i != j], 
                     lb=0, name=f"flow_{group_str}")

            # 根节点流出量 = 节点数 - 1
            model.addConstr(
                gp.quicksum(flow[root, j] for j in devices if j != root) == len(devices)-1,
                f"flow_root_{group_str}"
            )

            # 其他节点流入量 = 1
            for node in devices:
                if node == root:
                    continue
                model.addConstr(
                    gp.quicksum(flow[j, node] for j in devices if j != node) == 1,
                    f"flow_node_{node}_{group_str}"
                )

            # 流量约束: 流量 <= 边容量
            for i in devices:
                for j in devices:
                    if i == j:
                        continue
                    model.addConstr(flow[i,j] <= (len(devices)-1) * z.get((i,j), 0), 
                                f"flow_cap_{i}{j}_{group_str}")

            # 计算曼哈顿距离并线性化目标项
            for i, j in pairs:
                # 设备坐标的线性表达式
                xi = gp.quicksum(x * placement[i,x,y] for x in range(X) for y in range(Y))
                yi = gp.quicksum(y * placement[i,x,y] for x in range(X) for y in range(Y))
                xj = gp.quicksum(x * placement[j,x,y] for x in range(X) for y in range(Y))
                yj = gp.quicksum(y * placement[j,x,y] for x in range(X) for y in range(Y))

                # 曼哈顿距离 = |xi-xj| + |yi-yj|
                dx_plus = model.addVar(lb=0, name=f"dx+_{i}{j}")
                dx_minus = model.addVar(lb=0, name=f"dx-_{i}{j}")
                model.addConstr(xi - xj == dx_plus - dx_minus, f"dx_{i}{j}")
                
                dy_plus = model.addVar(lb=0, name=f"dy+_{i}{j}")
                dy_minus = model.addVar(lb=0, name=f"dy-_{i}{j}")
                model.addConstr(yi - yj == dy_plus - dy_minus, f"dy_{i}{j}")

                mh_dist = dx_plus + dx_minus + dy_plus + dy_minus

                # 线性化: cost = mh_dist * z[i,j]
                w = model.addVar(lb=0, name=f"w_{i}{j}")
                model.addConstr(w <= max_M * z[i,j], f"w_ub_{i}{j}")
                model.addConstr(w <= mh_dist, f"w_md_{i}{j}")
                model.addConstr(w >= mh_dist - max_M*(1 - z[i,j]), f"w_lb_{i}{j}")

                total_cost += freq * w

        model.setObjective(total_cost, GRB.MINIMIZE)
        
        model.Params.TimeLimit = time_limit  # 时间限制(秒)
        model.Params.MIPGap = 0.05           # 允许5%的优化间隙
        model.Params.Threads = 8             # 使用多线程
        model.Params.Heuristics = 0.1
        model.optimize()
        # 提取优化后的放置矩阵
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            optimized_placement = np.zeros((D, X, Y), dtype=int)
            for d in range(D):
                for x in range(X):
                    for y in range(Y):
                        if placement[d,x,y].X > 0.5:
                            optimized_placement[d,x,y] = 1

        return optimized_placement
    
    def optimize_placement_sa(self, initial_placement, P, layer_id, 
                            max_iter=1000, initial_temp=1000, cooling_rate=0.99):
        """
        使用模拟退火算法优化设备放置，最小化通信距离评估指标
        :param initial_placement: 初始设备放置矩阵 (D x X x Y)
        :param P: 固定的专家-设备分配矩阵 (E x D)
        :param layer_id: 当前层ID
        :param max_iter: 最大迭代次数
        :param initial_temp: 初始温度
        :param cooling_rate: 温度冷却率
        :return: 优化后的 placement 矩阵
        """
        current_placement = initial_placement.copy()
        current_cost = self.evaluate_placement(current_placement, P, layer_id)
        #current_cost,link = self.comm_time_acc(current_placement, P, layer_id)
        best_placement = current_placement.copy()
        best_cost = current_cost
        
        temp = initial_temp
        cost_history = [best_cost]
        
        for i in tqdm(range(max_iter)):
            # 生成候选解（随机扰动设备位置）
            new_placement = self._perturb_placement(current_placement)
            
            # 计算新解的成本
            new_cost = self.evaluate_placement(new_placement, P, layer_id)
            #new_cost,link = self.comm_time_acc(new_placement, P, layer_id)
            # 计算成本差
            cost_diff = new_cost - current_cost
            
            # 接受条件：如果新解更优，或满足概率条件
            if cost_diff < 0 or math.exp(-cost_diff / temp) > random.random():
                current_placement = new_placement.copy()
                current_cost = new_cost
                
                # 更新历史最优解
                if new_cost < best_cost:
                    best_placement = new_placement.copy()
                    best_cost = new_cost
            #pdb.set_trace()
            cost_history.append(float(best_cost)) 
            # 降低温度
            temp *= cooling_rate
        
        return best_placement, cost_history

    def _perturb_placement(self, placement):
        """
        随机交换两个设备的位置
        """
        new_placement = placement.copy()
        D = new_placement.shape[0]
        
        # 随机选择两个设备
        d1, d2 = np.random.choice(D, 2, replace=False)
        
        # 交换它们的位置
        pos1 = np.argwhere(new_placement[d1] == 1)[0]
        pos2 = np.argwhere(new_placement[d2] == 1)[0]
        
        new_placement[d1, pos1[0], pos1[1]] = 0
        new_placement[d2, pos2[0], pos2[1]] = 0
        new_placement[d1, pos2[0], pos2[1]] = 1
        new_placement[d2, pos1[0], pos1[1]] = 1
        
        return new_placement


    def evaluate_placement(self, placement, P,layer_id):
        """
        基于最小生成树的通信距离评估
        :param placement: D x X x Y的放置矩阵
        :param P: E x D的专家-设备分配矩阵
        :return: 加权平均通信距离（MST总距离）
        """
        X, Y = placement.shape[1], placement.shape[2]
        device_coords = {d: np.argwhere(placement[d] == 1)[0] for d in range(self.D)}
        expert_to_device = [np.where(P[layer_id][e] == 1)[0].tolist() for e in range(self.E)]
        
        total_weight = 0.0
        total_freq = 0.0
        

  
        for group_str, freq in self.fg[self.e][layer_id].items():
            devices = [expert_to_device[e] for e in ast.literal_eval(group_str)]
            devices = list(set(d for sublist in devices for d in sublist))
            coords = [tuple(device_coords[d]) for d in devices]
            
            if len(coords) < 2:
                continue
                
            # 计算MST总距离
            mst_dist = self._calculate_mst(coords)
            total_weight += freq * mst_dist
            total_freq += freq
                    
        return total_weight / total_freq if total_freq > 0 else 0

    def _calculate_mst(self, coords):
        """ Kruskal算法计算最小生成树 """
        edges = []
        n = len(coords)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(coords[i][0] - coords[j][0])
                dy = abs(coords[i][1] - coords[j][1])
                edges.append((i, j, dx + dy))
        
        edges.sort(key=lambda x: x[2])
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            parent[find(u)] = find(v)
        
        mst_sum = 0
        for u, v, w in edges:
            if find(u) != find(v):
                union(u, v)
                mst_sum += w
        return mst_sum
    def optimize_placement_bo(self, initial_placement, P, layer_id, 
                             max_iter=50, random_state=None):
        """
        使用贝叶斯优化算法优化设备放置，最小化通信距离评估指标
        :param initial_placement: 初始设备放置矩阵 (D x X x Y)
        :param P: 固定的专家-设备分配矩阵 (E x D)
        :param layer_id: 当前层ID
        :param max_iter: 最大迭代次数
        :param random_state: 随机种子
        :return: 优化后的 placement 矩阵和成本历史
        """
        D, X, Y = initial_placement.shape
        assert D == X * Y, "设备数量必须等于网格位置数"
        #np.set_printoptions(threshold=np.inf)
        #print(P[layer_id])
        # 提取初始设备的坐标作为贝叶斯优化的初始点
        initial_params = []
        for d in range(D):
            pos = np.argwhere(initial_placement[d] == 1)[0]
            initial_params.extend(pos.tolist())  # 转换为连续参数
            
        # 定义参数空间：每个设备的x和y坐标范围
        space = [Real(0, X-1), Real(0, Y-1)] * D
        
        # 定义目标函数（通过闭包捕获self, P, layer_id等参数）
        def objective(params):
            device_coords = np.array(params).reshape(D, 2)
            grid_points = np.array([(x, y) for x in range(X) for y in range(Y)])
            cost_matrix = np.zeros((D, X*Y))
            
            # 计算曼哈顿距离成本矩阵
            for d in range(D):
                for g, (x, y) in enumerate(grid_points):
                    dx = abs(device_coords[d, 0] - x)
                    dy = abs(device_coords[d, 1] - y)
                    cost_matrix[d, g] = dx + dy
            
            # 使用匈牙利算法进行最优分配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 构建合法的placement矩阵
            placement = np.zeros((D, X, Y), dtype=int)
            for d in range(D):
                x, y = grid_points[col_ind[d]]
                placement[d, x, y] = 1
            result,link=self.comm_time_acc(placement, P, layer_id)
            #result=self.evaluate_placement(placement, P, layer_id)
            #print(result)
            return result
        
        # 运行贝叶斯优化
        result = gp_minimize(
            objective,
            space,
            n_calls=max_iter,
            x0=initial_params,
            random_state=random_state,
            n_initial_points=min(10, max_iter),  # 初始评估点数
            verbose=True
        )
        
        # 获取最优参数并生成最终placement矩阵
        best_params = result.x
        grid_points = np.array([(x, y) for x in range(X) for y in range(Y)])
        device_coords = np.array(best_params).reshape(D, 2)
        cost_matrix = np.zeros((D, X*Y))
        for d in range(D):
            for g, (x, y) in enumerate(grid_points):
                dx = abs(device_coords[d, 0] - x)
                dy = abs(device_coords[d, 1] - y)
                cost_matrix[d, g] = dx + dy
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        best_placement = np.zeros((D, X, Y), dtype=int)
        for d in range(D):
            x, y = grid_points[col_ind[d]]
            best_placement[d, x, y] = 1
        
        return best_placement, result.func_vals
    # ----------------- 动态部署策略 -----------------
    def priority_detection(self, P,layer_id,random_samples):
        """优先级检测 (公式9)"""
        if layer_id>0:

            #node_load = np.sum(P * self.f[layer_id-1, None] * 2 * self.h*self.IS / self.comp, axis=0)
            #node_load = np.sum(P * self.f[:,:, None] * self.B * 2 * self.h*self.IS, axis=1)
            
            comp_map=np.zeros((self.E))
            
            for sublist in random_samples:
                comp_map[sublist]+=2*self.h*self.IS
            node_load = np.sum(P * comp_map[None,:,None], axis=1)
            congested_node = np.argmax(node_load,axis=1)[layer_id]
            priorities = []
            for i in range(self.E):
                prio = (P[layer_id,i, congested_node] * comp_map[i] / self.comp)
                priorities.append((prio, i))
        #pdb.set_trace()
        return sorted(priorities, reverse=True)[0]





    def optimal_broadcast_chunk(self, alpha=1e-7, k=1):
        """α-β最优广播块 (公式10)"""
        beta=1/self.BW
        c =  np.sqrt(2*self.h*self.IS*alpha / (2*beta * k * np.sqrt(self.D)))
        latency = alpha * (2 * np.sqrt(self.D) + 2*self.h*self.IS / c)
        bandwidth = beta * k * (2*self.h*self.IS + 2 * c * np.sqrt(self.D))
        return latency + bandwidth





