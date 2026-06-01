# lpa.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)

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
# 1. LPA function
# ---------------------------------------------------------------------------#
def label_propagation_community_detection(G, max_iter):
    labels = {node: i for i, node in enumerate(G.nodes())}
    
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    
    actual_iterations = 0
    
    for iteration in range(max_iter):
        changed = False
        for node in nodes:
            if G.degree(node) == 0:
                continue
                
            neighbor_labels = {}
            for neighbor in G.neighbors(node):
                label = labels[neighbor]
                neighbor_labels[label] = neighbor_labels.get(label, 0) + 1
            
            if not neighbor_labels:
                continue
                
            max_count = max(neighbor_labels.values())
            most_common_labels = [label for label, count in neighbor_labels.items() 
                                  if count == max_count]
            
            if labels[node] not in most_common_labels:
                labels[node] = np.random.choice(most_common_labels)
                changed = True
        
        actual_iterations = iteration + 1
        
        if not changed:
            print(f"  LPA converged at iteration {iteration+1} (early termination)")
            break
    
    return labels, actual_iterations

# ---------------------------------------------------------------------------#
# 2. evaluation modularity                                                                
# ---------------------------------------------------------------------------#
def compute_modularity(partition, G):
    m = G.number_of_edges()
    if m == 0:
        return 0
    
    community_to_nodes = {}
    for node, comm in partition.items():
        community_to_nodes.setdefault(comm, []).append(node)
    
    modularity = 0
    for comm, nodes_in_comm in community_to_nodes.items():
        L_c = 0
        for i in range(len(nodes_in_comm)):
            for j in range(i+1, len(nodes_in_comm)):
                if G.has_edge(nodes_in_comm[i], nodes_in_comm[j]):
                    L_c += 1
        
        d_c = 0
        for node in nodes_in_comm:
            d_c += G.degree(node)
        
        modularity += (L_c / m) - (d_c / (2 * m)) ** 2
    
    return modularity

# ---------------------------------------------------------------------------#
# 3. Example entry
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    LPA_MAX_ITER = 100           
    RANDOM_SEED = 42           

    file_name = "tree" 
    # file_name = "lol" 
    # file_name = "LFR_base" 
    # file_name = "email-Eu-core" 
    # file_name = "facebook" 
    # file_name = "com-youtube_largest_deliso" 

    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    np.random.seed(RANDOM_SEED)
    G = load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]
    
    partition, actual_iterations = label_propagation_community_detection(G, max_iter=LPA_MAX_ITER)
    
    pred_labels = [partition[n] for n in player_names]
    
    modularity = compute_modularity(partition, G)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print(f"{file_name} lpa_result: Modularity: {modularity:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
