#fastgreedy.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import modularity

# ---------------------------------------------------------------------------#
#  0.   read network
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
#  1.   Example entry
# ---------------------------------------------------------------------------#
if __name__ == "__main__":

    #tree 
    file_name = "tree"  
    input_dir = os.path.join('..', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    # lol
    # file_name = "lol" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    # LFR_base
    # file_name = "LFR_base" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    # email-Eu-core
    # file_name = "email-Eu-core" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    # facebook
    # file_name = "facebook" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')

    # com-youtube_largest_deliso
    # file_name = "com-youtube_largest_deliso" 
    # input_dir = os.path.join('..', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    
    G = load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]
    
    communities = list(greedy_modularity_communities(G))
    
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    
    pred_labels = [partition[n] for n in player_names]
    
    modularity_score = modularity(G, communities)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    print(f"{file_name} fastgreedy_result: modularity: {modularity_score:.6f}, ARI: {ari:.6f}, NMI: {nmi:.6f}")
