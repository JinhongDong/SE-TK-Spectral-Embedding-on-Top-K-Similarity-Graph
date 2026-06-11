#fastgreedy.py
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
from heapq import heappush, heappop, heapify
from networkx.algorithms.community import greedy_modularity_communities
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------#
# 0. Load network
# ---------------------------------------------------------------------------#
def load_graph_with_attributes(node_file_path, edge_file_path):
    G = nx.Graph()
    nodes_data = {}
    with open(node_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, comm = parts
                nodes_data[int(node_id)] = {'actual_community': int(comm)}
    G.add_nodes_from(nodes_data.items())
    edges = []
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                n1, n2 = parts
                edges.append((int(n1), int(n2)))
    G.add_edges_from(edges)
    return G

# ---------------------------------------------------------------------------#
# 1. fastgreedy function
# ---------------------------------------------------------------------------#
def fastgreedy_memory_efficient(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    node_list = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    adj_list = defaultdict(list)
    degrees = np.zeros(n, dtype=np.float32)
    
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj_list[i].append(j)
        adj_list[j].append(i)
        degrees[i] += 1
        degrees[j] += 1
    
    community = np.arange(n, dtype=np.int32)
    community_degree = degrees.copy()  
    community_size = np.ones(n, dtype=np.int32)  
    
    A_inter = {}  
    for i in range(n):
        for j in adj_list[i]:
            if i < j:  
                c1, c2 = community[i], community[j]
                if c1 != c2:
                    key = (min(c1, c2), max(c1, c2))
                    A_inter[key] = A_inter.get(key, 0.0) + 1.0
    
    m_float = float(m)
    heap = []
    
    for (c1, c2), A_ij in A_inter.items():
        delta_Q = A_ij / m_float - (community_degree[c1] * community_degree[c2]) / (2 * m_float * m_float)
        heappush(heap, (-delta_Q, c1, c2, A_ij))
    
    active_communities = set(range(n))
    merge_history = []
    
    while heap and len(active_communities) > 1:
        neg_delta_Q, c1, c2, A_ij = heappop(heap)
        delta_Q = -neg_delta_Q
        if c1 not in active_communities or c2 not in active_communities:
            continue
        if delta_Q <= 1e-7:  
            break
        
        merge_history.append((c1, c2, delta_Q))

        active_communities.remove(c2)
        mask = (community == c2)
        community[mask] = c1

        community_degree[c1] += community_degree[c2]
        community_size[c1] += community_size[c2]
        community_degree[c2] = 0
        community_size[c2] = 0

        new_edges = {}
        related_edges = defaultdict(float)

        for (ci, cj), weight in list(A_inter.items()):
            if ci == c2 or cj == c2:
                other = ci if cj == c2 else cj
                if other == c1:
                    continue
                new_c1, new_c2 = min(c1, other), max(c1, other)
                related_edges[(new_c1, new_c2)] += weight
            elif ci == c1 or cj == c1:
                other = ci if cj == c1 else cj
                if other in active_communities:
                    related_edges[(min(c1, other), max(c1, other))] += weight

        A_inter = {k: v for k, v in A_inter.items() 
                  if k[0] != c1 and k[1] != c1 and k[0] != c2 and k[1] != c2}
        
        for (ci, cj), weight in related_edges.items():
            A_inter[(ci, cj)] = weight
        
        new_heap = []
        for item in heap:
            neg_q, ci, cj, A = item
            if ci != c1 and ci != c2 and cj != c1 and cj != c2:
                new_heap.append(item)
        heap = new_heap
        heapify(heap)
        
        for (ci, cj), A_ij in A_inter.items():
            if ci == c1 or cj == c1:
                delta_Q = A_ij / m_float - (community_degree[c1] * community_degree[cj if ci == c1 else ci]) / (2 * m_float * m_float)
                heappush(heap, (-delta_Q, ci, cj, A_ij))
        
        if len(active_communities) % 1000 == 0 or len(active_communities) <= 10:
            print(f"Remaining number of communities: {len(active_communities)}, now ΔQ: {delta_Q:.6f}")
    
    print(f"Merging completed, final number of communities: {len(active_communities)}")
    
    unique_comms = np.unique(community)
    comm_mapping = {old: new for new, old in enumerate(unique_comms)}
    
    partition = {}
    for idx, node in enumerate(node_list):
        comm_id = comm_mapping[community[idx]]
        partition[node] = comm_id
    
    return partition

def fastgreedy_auto(G):
    n = G.number_of_nodes()
    if n >= 10000:
        return fastgreedy_memory_efficient(G)
    else:
        communities = list(greedy_modularity_communities(G))
        partition = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = cid
        return partition

# ---------------------------------------------------------------------------#
# 2. Example entry
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    
    file_name="tree"
    # file_name = 'LFR_base'
    # file_name = 'lol'
    # file_name = 'email-Eu-core'
    # file_name = 'facebook'
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
