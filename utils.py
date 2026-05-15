import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
import community as community_louvain
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import eigh
from sklearn.model_selection import ParameterGrid
import warnings
import random

# ---------- Utility Functions ----------
def load_graph_with_attributes(node_file_path, edge_file_path):
    """Load graph with attributes, time complexity O(V+E)"""
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

def best_map(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    D = max(pred_labels.max(), true_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred_labels.size):
        w[pred_labels[i], true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in pred_labels])

def enhanced_structural_embeddings(G, n_components,player_names):
    """Spectral embedding method, time complexity O(V^3)"""
    
    # Spectral embedding method
    adj_matrix = nx.to_numpy_array(G, nodelist=player_names)  # Build adjacency matrix
    degrees = np.array([d for _, d in G.degree(player_names)])  # Compute degree matrix
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))  # Compute degree inverse square root matrix
    laplacian = np.eye(len(player_names)) - deg_inv_sqrt @ adj_matrix @ deg_inv_sqrt  # Build Laplacian matrix
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian)
    idx = eigenvalues.argsort()
    
    # Select smallest eigenvectors with eigenvalues greater than zero
    start_idx = np.sum(eigenvalues < 1e-8)
    end_idx = min(start_idx + n_components, len(player_names))
    
    embeddings = eigenvectors[:, idx[start_idx:end_idx]]
    
    # Normalize embedding vectors
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return embeddings

def community_topk_similarity_graph(G, embeddings, player_names, resolution,
                                   similarity_threshold, preserve_ratio,
                                   max_preserved_edges, k_factor):
    """Build new graph based on community structure and TopK similarity, time complexity O(V^2 + E)"""
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix = np.maximum(similarity_matrix, 0)
    G_emb = nx.Graph()
    G_emb.add_nodes_from(player_names)
    
    # 1. Preserve original edges based on community structure (preserve intra-community edges)
    try:
        # Use Louvain to detect original community structure
        comm = community_louvain.best_partition(G, resolution=resolution)
        intra_edges = []
        for u, v in G.edges():
            if comm[u] == comm[v]:
                intra_edges.append((u, v))
        
        max_edges = min(max_preserved_edges, len(intra_edges))
        num_to_preserve = min(max_edges, int(len(intra_edges) * preserve_ratio))
        
        # Randomly select intra-community edges to preserve
        random.shuffle(intra_edges)
        for u, v in intra_edges[:num_to_preserve]:
            G_emb.add_edge(u, v, weight=1.0, type="preserved")
    except Exception as e:
        print(f"Community structure preservation failed: {e}")
        # If community detection fails, use centrality strategy as backup
        try:
            edge_centrality = nx.edge_betweenness_centrality(G)
            sorted_edges = sorted(G.edges(), key=lambda x: edge_centrality[x], reverse=True)
            for u, v in sorted_edges[:int(len(sorted_edges)*preserve_ratio)]:
                G_emb.add_edge(u, v, weight=1.0, type="preserved")
        except:
            # If centrality calculation fails, randomly preserve some edges
            edges = list(G.edges())
            random.shuffle(edges)
            for u, v in edges[:min(100, len(edges))]:
                G_emb.add_edge(u, v, weight=1.0, type="preserved")
    
    # 2. Add TopK similarity edges
    k = min(k_factor, max(1, len(player_names)//10))  
    for i, u in enumerate(player_names):
        # Get k nodes with highest similarity (excluding self)
        sim_scores = similarity_matrix[i].copy()
        sim_scores[i] = -1  
        top_indices = np.argsort(sim_scores)[::-1][:k]
        
        for idx in top_indices:
            if sim_scores[idx] > similarity_threshold:  
                v = player_names[idx]
                if not G_emb.has_edge(u, v):
                    G_emb.add_edge(u, v, weight=sim_scores[idx], type="added")
    
    # 3. Ensure minimum connectivity
    for node in player_names:
        if G_emb.degree(node) < 2:
            i = player_names.index(node)
            sims = similarity_matrix[i].copy()
            sims[i] = -1
            most_similar = np.argmax(sims)
            if sims[most_similar] > similarity_threshold:
                v = player_names[most_similar]
                if not G_emb.has_edge(node, v):
                    G_emb.add_edge(node, v, weight=sims[most_similar], type="connectivity")
    
    return G_emb

def louvain_community_detection(G, resolution, n_iter, use_weight):
    """Louvain community detection method, time complexity O(n_iter * V log V)"""
    best_partition = None
    best_modularity = -1
    for _ in range(n_iter):  # Multiple runs, take the best
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
        mod = community_louvain.modularity(partition, G, weight='weight')
        if mod > best_modularity:
            best_modularity = mod
            best_partition = partition
    return best_partition

def hierarchical_community_optimization(G, partition, min_size, 
                                       size_ratio):
    """Hierarchical community optimization, Merge overly small communities, time complexity O(V + E)"""

    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)
    
    # Calculate average community size
    sizes = [len(nodes) for nodes in comm_nodes.values()]
    if sizes:
        avg_size = np.mean(sizes)
        min_size = max(min_size, int(avg_size * size_ratio))
    else:
        min_size = max(min_size, 2)
    
    small_comms = [comm for comm, nodes in comm_nodes.items() if len(nodes) < min_size]
    for comm in small_comms:
        # Find the most similar neighboring community
        neighbor_comms = defaultdict(float)
        for node in comm_nodes[comm]:
            for neighbor in G.neighbors(node):
                neighbor_comm = partition[neighbor]
                if neighbor_comm != comm:
                    weight = G[node][neighbor].get('weight', 1.0)
                    neighbor_comms[neighbor_comm] += weight
        
        if neighbor_comms:
            target_comm = max(neighbor_comms, key=neighbor_comms.get)
            for node in comm_nodes[comm]:
                partition[node] = target_comm
        else:
            # If no neighboring communities, randomly assign to a large community
            large_comms = [c for c, nodes in comm_nodes.items() if len(nodes) >= min_size]
            if large_comms:
                target_comm = random.choice(large_comms)
                for node in comm_nodes[comm]:
                    partition[node] = target_comm
    
    return partition
