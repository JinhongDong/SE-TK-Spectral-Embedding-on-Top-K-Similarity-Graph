#louvain.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)
import community as community_louvain

# ---------------------------------------------------------------------------#
# 0. Read network
# ---------------------------------------------------------------------------#
def load_graph_with_attributes(node_file_path, edge_file_path):
    G = nx.Graph()
    with open(node_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node_id, comm = parts
                G.add_node(int(node_id), actual_community=int(comm))
    with open(edge_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                n1, n2 = parts
                G.add_edge(int(n1), int(n2))
    return G

# ---------------------------------------------------------------------------#
# 1. Example entry
# ---------------------------------------------------------------------------#
if __name__ == "__main__":

    file_name = "tree"  
    # file_name = "lol" 
    # file_name = "LFR_base" 
    # file_name = "email-Eu-core" 
    # file_name = "facebook" 
    # file_name = "com-youtube_largest_deliso" 

    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    
    resolution=1.0
    G = load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]

    partition = community_louvain.best_partition(G, resolution=resolution)
    
    pred_labels = [partition[n] for n in player_names]
    
    modularity = community_louvain.modularity(partition, G)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print(f"{file_name} louvain_result:   modularity: {modularity:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
