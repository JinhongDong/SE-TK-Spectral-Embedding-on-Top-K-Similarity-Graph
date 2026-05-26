import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy import sparse
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score
)
from collections import defaultdict
import heapq
from itertools import combinations
import warnings
from networkx.algorithms.community import modularity
warnings.filterwarnings('ignore')

def load_graph_with_attributes(node_file_path, edge_file_path):
    """加载带属性的图 - 优化版本"""
    G = nx.Graph()
    
    # 批量加载节点
    nodes_data = {}
    with open(node_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, comm = parts
                nodes_data[int(node_id)] = {'actual_community': int(comm)}
    
    # 批量添加节点
    G.add_nodes_from(nodes_data.items())
    
    # 批量加载边
    edges = []
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                n1, n2 = parts
                edges.append((int(n1), int(n2)))
    
    # 批量添加边
    G.add_edges_from(edges)
    
    return G


def fastgreedy_memory_efficient(G):
    """
    内存高效的FastGreedy实现
    使用增量合并和优先队列优化
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print(f"FastGreedy优化版本: {n}个节点, {m}条边")
    start_time = time.time()
    
    # 1. 节点到索引的映射
    node_list = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # 2. 构建邻接表（稀疏表示）
    adj_list = defaultdict(list)
    degrees = np.zeros(n, dtype=np.float32)
    
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj_list[i].append(j)
        adj_list[j].append(i)
        degrees[i] += 1
        degrees[j] += 1
    
    # 3. 初始化：每个节点一个社区
    community = np.arange(n, dtype=np.int32)
    community_degree = degrees.copy()  # 社区的度
    community_size = np.ones(n, dtype=np.int32)  # 社区大小
    
    # 4. 社区间的边数
    # 使用字典存储稀疏的社区间连接
    A_inter = {}  # 社区间的边权重
    for i in range(n):
        for j in adj_list[i]:
            if i < j:  # 避免重复
                c1, c2 = community[i], community[j]
                if c1 != c2:
                    key = (min(c1, c2), max(c1, c2))
                    A_inter[key] = A_inter.get(key, 0.0) + 1.0
    
    # 5. 优先队列存储所有可能的合并
    # 格式: (-delta_Q, c1, c2, A_inter[c1,c2])
    from heapq import heappush, heappop, heapify
    
    # 计算初始delta_Q
    m_float = float(m)
    heap = []
    
    for (c1, c2), A_ij in A_inter.items():
        delta_Q = A_ij / m_float - (community_degree[c1] * community_degree[c2]) / (2 * m_float * m_float)
        heappush(heap, (-delta_Q, c1, c2, A_ij))
    
    # 6. 增量合并
    active_communities = set(range(n))
    merge_history = []
    
    while heap and len(active_communities) > 1:
        # 获取最佳合并
        neg_delta_Q, c1, c2, A_ij = heappop(heap)
        delta_Q = -neg_delta_Q
        
        # 检查社区是否仍然存在
        if c1 not in active_communities or c2 not in active_communities:
            continue
        
        if delta_Q <= 1e-7:  # 增益太小，停止
            break
        
        # 记录合并
        merge_history.append((c1, c2, delta_Q))
        
        # 合并c2到c1
        active_communities.remove(c2)
        mask = (community == c2)
        community[mask] = c1
        
        # 更新社区度
        community_degree[c1] += community_degree[c2]
        community_size[c1] += community_size[c2]
        community_degree[c2] = 0
        community_size[c2] = 0
        
        # 更新社区间连接
        new_edges = {}
        
        # 收集与c1和c2相关的边
        related_edges = defaultdict(float)
        
        # 遍历所有与c1和c2相关的边
        for (ci, cj), weight in list(A_inter.items()):
            if ci == c2 or cj == c2:
                # 这条边与c2相关
                other = ci if cj == c2 else cj
                if other == c1:
                    # c1和c2之间的边变成自环，忽略
                    continue
                new_c1, new_c2 = min(c1, other), max(c1, other)
                related_edges[(new_c1, new_c2)] += weight
            elif ci == c1 or cj == c1:
                # 这条边与c1相关
                other = ci if cj == c1 else cj
                if other in active_communities:
                    related_edges[(min(c1, other), max(c1, other))] += weight
        
        # 从A_inter中移除与c1和c2相关的旧边
        A_inter = {k: v for k, v in A_inter.items() 
                  if k[0] != c1 and k[1] != c1 and k[0] != c2 and k[1] != c2}
        
        # 添加新的边
        for (ci, cj), weight in related_edges.items():
            A_inter[(ci, cj)] = weight
        
        # 重新计算与c1相关的delta_Q并更新堆
        # 首先移除堆中与c1和c2相关的旧条目
        new_heap = []
        for item in heap:
            neg_q, ci, cj, A = item
            if ci != c1 and ci != c2 and cj != c1 and cj != c2:
                new_heap.append(item)
        heap = new_heap
        heapify(heap)
        
        # 添加与c1相关的新条目
        for (ci, cj), A_ij in A_inter.items():
            if ci == c1 or cj == c1:
                delta_Q = A_ij / m_float - (community_degree[c1] * community_degree[cj if ci == c1 else ci]) / (2 * m_float * m_float)
                heappush(heap, (-delta_Q, ci, cj, A_ij))
        
        if len(active_communities) % 1000 == 0 or len(active_communities) <= 10:
            print(f"剩余社区数: {len(active_communities)}, 当前ΔQ: {delta_Q:.6f}")
    
    print(f"合并完成，最终社区数: {len(active_communities)}, 耗时: {time.time()-start_time:.2f}秒")
    
    # 7. 重建社区分配
    unique_comms = np.unique(community)
    comm_mapping = {old: new for new, old in enumerate(unique_comms)}
    
    partition = {}
    for idx, node in enumerate(node_list):
        comm_id = comm_mapping[community[idx]]
        partition[node] = comm_id
    
    return partition

def fastgreedy_sparse_matrix(G):
    """
    使用稀疏矩阵的FastGreedy实现
    适合大规模图
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print(f"稀疏矩阵FastGreedy: {n}个节点, {m}条边")
    start_time = time.time()
    
    # 1. 节点到索引的映射
    node_list = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 2. 构建稀疏邻接矩阵
    from scipy.sparse import lil_matrix, csr_matrix
    
    A = lil_matrix((n, n), dtype=np.float32)
    degrees = np.zeros(n, dtype=np.float32)
    
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        A[i, j] = 1
        A[j, i] = 1
        degrees[i] += 1
        degrees[j] += 1
    
    A = A.tocsr()
    
    # 3. 使用NetworkX的优化实现
    print("调用NetworkX优化FastGreedy...")
    communities = list(greedy_modularity_communities(G))
    
    # 4. 转换为分区格式
    partition = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            partition[node] = comm_id
    
    print(f"计算完成，耗时: {time.time()-start_time:.2f}秒")
    return partition



def fastgreedy_gpu_fastest(G):
    """GPU加速版本 - 使用NetworkX算法，GPU加速后续计算"""
    from networkx.algorithms.community import greedy_modularity_communities
    
    print("运行NetworkX FastGreedy算法（C++后端）...")
    start_time = time.time()
    communities = list(greedy_modularity_communities(G))
    print(f"社区检测耗时: {time.time()-start_time:.2f}秒")
    
    # GPU加速社区分配
    print("GPU加速社区分配...")
    node_list = sorted(G.nodes())
    
    if torch.cuda.is_available():
        # 使用GPU批量处理
        node_tensor = torch.tensor(node_list, dtype=torch.long, device='cuda')
        
        # 预先分配结果
        partition_tensor = -torch.ones(len(node_list), dtype=torch.long, device='cuda')
        
        for i, comm in enumerate(communities):
            if len(comm) > 0:
                comm_tensor = torch.tensor(list(comm), dtype=torch.long, device='cuda')
                # 向量化查找
                mask = (node_tensor.unsqueeze(1) == comm_tensor.unsqueeze(0)).any(dim=1)
                partition_tensor[mask] = i
        
        # 转换结果
        partition_cpu = partition_tensor.cpu().numpy()
        partition = {node: int(partition_cpu[idx]) for idx, node in enumerate(node_list)}
    else:
        # CPU版本
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
    
    return partition

def fastgreedy_auto(G):
    n = G.number_of_nodes()
    if n >= 10000:
        return fastgreedy_memory_efficient(G)

    else:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = cid
        return partition



if __name__ == "__main__":
    
    # file_name="tree"
    # file_name = 'LFR_base'
    # file_name = 'lol'
    # file_name = 'email-Eu-core'
    file_name = 'facebook'
    # file_name = 'com-youtube_largest_deliso'
    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    
    G = load_graph_with_attributes(node_file_path, edge_file_path)
    
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]
    
    partition = fastgreedy_auto(G)

    pred_labels = [partition[n] for n in player_names]

    comm_dict = {}
    for node, comm in partition.items():
        if comm not in comm_dict:
            comm_dict[comm] = []
        comm_dict[comm].append(node)
    communities_list = list(comm_dict.values())
    
    modularity_score = modularity(G, communities_list)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    
    print(f"{file_name} fastgreedy_result: Modularity: {modularity_score:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
