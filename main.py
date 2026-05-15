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
import argparse
import utils  

if __name__ == "__main__":
    

    #tree 
    file_name = "tree"  
    input_dir = os.path.join('.', 'norm_dataset', file_name)
    node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    embed_n_components = 20
    embed_use_adaptive = True  
    graph_similarity_threshold = 0.5
    graph_preserve_ratio = 0.4
    graph_max_preserved_edges = 500
    graph_k_factor = 10  
    louvain_n_iter = 10
    louvain_resolution = 2.0
    louvain_use_weight = True
    optimize_min_size = 3
    optimize_size_ratio = 0.3
    optimize_merge_small = True
    use_grid_search = True

    # lol
    # file_name = "lol"  
    # input_dir = os.path.join('.', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # embed_n_components = 10
    # embed_use_adaptive = True  
    # graph_similarity_threshold = 0.3
    # graph_preserve_ratio = 0.4
    # graph_max_preserved_edges = 500
    # graph_k_factor = 5  
    # louvain_n_iter = 5
    # louvain_resolution = 2.0
    # louvain_use_weight = True
    # optimize_min_size = 3
    # optimize_size_ratio = 0.2
    # optimize_merge_small = True
    # use_grid_search = True

    # LFR_base
    # file_name = "LFR_base"  
    # input_dir = os.path.join('.', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # embed_n_components = 30
    # embed_use_adaptive = True  
    # graph_similarity_threshold = 0.7
    # graph_preserve_ratio = 0.8
    # graph_max_preserved_edges = 1000
    # graph_k_factor = 10  
    # louvain_n_iter = 10
    # louvain_resolution = 1.2
    # louvain_use_weight = True
    # optimize_min_size = 3
    # optimize_size_ratio = 0.2
    # optimize_merge_small = True
    # use_grid_search = True

    # email-Eu-core
    # file_name = "email-Eu-core"  
    # input_dir = os.path.join('.', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # embed_n_components = 90
    # embed_use_adaptive = True  
    # graph_similarity_threshold = 0.3
    # graph_preserve_ratio = 0.4
    # graph_max_preserved_edges = 500
    # graph_k_factor = 10  
    # louvain_n_iter = 5
    # louvain_resolution = 1.5
    # louvain_use_weight = True
    # optimize_min_size = 3
    # optimize_size_ratio = 0.2
    # optimize_merge_small = True
    # use_grid_search = True

    # facebook
    # file_name = "facebook"  
    # input_dir = os.path.join('.', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # embed_n_components = 50
    # embed_use_adaptive = True  
    # graph_similarity_threshold = 0.7
    # graph_preserve_ratio = 0.5
    # graph_max_preserved_edges = 3000
    # graph_k_factor = 15  
    # louvain_n_iter = 5
    # louvain_resolution = 1.2
    # louvain_use_weight = True
    # optimize_min_size = 10
    # optimize_size_ratio = 0.2
    # optimize_merge_small = True
    # use_grid_search = True

    # com-youtube_largest_deliso
    # file_name = "com-youtube_largest_deliso"  
    # input_dir = os.path.join('.', 'norm_dataset', file_name)
    # node_file_path = os.path.join(input_dir, f'{file_name}_nodes.txt')
    # edge_file_path = os.path.join(input_dir, f'{file_name}_edges.txt')
    # embed_n_components = 256
    # embed_use_adaptive = True  
    # graph_similarity_threshold = 0.5
    # graph_preserve_ratio = 0.4
    # graph_max_preserved_edges = 10000
    # graph_k_factor = 20  
    # louvain_n_iter = 5
    # louvain_resolution = 0.6
    # louvain_use_weight = True
    # optimize_min_size = 10
    # optimize_size_ratio = 0.4
    # optimize_merge_small = True
    # use_grid_search = True

    composite_score_weights = {
        'ARI': 0.5,
        'NMI': 0.3,
        'modularity_original_graph': 0.2
    }
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    G = utils.load_graph_with_attributes(node_file_path, edge_file_path)
    player_names = sorted(G.nodes())
    true_labels = [G.nodes[n]['actual_community'] for n in player_names]
    
    V = len(player_names)  
    E = G.number_of_edges()  
    density = 2*E / (V*(V-1)) if V>1 else 0
    
    if use_grid_search:
        if V <= 100:  # Small graph
            param_grid = {
                'n_components': [10, 15, 20, 30],
                'resolution': [0.5, 0.8, 1.0, 1.5, 2.0]
            }
        elif V <= 1000:  # Medium graph
            if density > 0.1:  # Dense
                param_grid = {
                    'n_components': [30, 50, 70, 100],
                    'resolution': [0.8, 1.0, 1.2, 1.5, 2.0]
                }
            else:  # Sparse
                param_grid = {
                    'n_components': [20, 30, 50, 70],
                    'resolution': [0.5, 0.8, 1.0, 1.2,1.5]
                }
        elif V <= 10000: 
            param_grid = {
                    'n_components': [50, 70, 90,100],
                    'resolution': [0.8, 1.0, 1.2, 1.5, 2.0]
                }
        else:  # Large graph
            param_grid = {
                'n_components': [256, 512, 1024],
                'resolution': [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
            }
    else:
        # Do not use grid search, use fixed parameters
        param_grid = {
            'n_components': [embed_n_components],
            'resolution': [louvain_resolution]
        }

    best_score = -1
    best_result = {}
    
    for idx, params in enumerate(ParameterGrid(param_grid)):
        try:
            # 1. Generate spectral embeddings
            embeddings = utils.enhanced_structural_embeddings(G, params['n_components'],player_names)
            
            # 2. Build new graph based on community structure and TopK similarity
            G_emb = utils.community_topk_similarity_graph(
                G, embeddings, player_names, 
                resolution=params['resolution'],
                similarity_threshold=graph_similarity_threshold,
                preserve_ratio=graph_preserve_ratio,
                max_preserved_edges=graph_max_preserved_edges,
                k_factor=graph_k_factor
            )
            
            # 3. Louvain community detection
            partition = utils.louvain_community_detection(
                G_emb, 
                resolution=params['resolution'],
                n_iter=louvain_n_iter,
                use_weight=louvain_use_weight
            )
            
            # 4. Community optimization
            partition = utils.hierarchical_community_optimization(
                G_emb, partition,
                min_size=optimize_min_size,
                size_ratio=optimize_size_ratio,
                merge_small=optimize_merge_small
            )
            
            pred_labels = [partition[n] for n in player_names]
            pred_labels = utils.best_map(true_labels, pred_labels)

            # Evaluation metrics
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)
            
            try:
                modularity_original_graph = community_louvain.modularity(partition, G)
                modularity_similarity_graph = community_louvain.modularity(partition, G_emb)
            except:
                modularity_original_graph = 0.0
                modularity_similarity_graph = 0.0
                
            # Calculate edge retention rate
            orig_edges = set(G.edges())
            new_edges = set(G_emb.edges())
            edge_keep = len(orig_edges & new_edges) / len(orig_edges) if orig_edges else 0.0
            
            metrics = {
                'NMI': nmi,
                'ARI': ari,
                'modularity_original_graph': modularity_original_graph,
                'modularity_similarity_graph': modularity_similarity_graph,
                'EdgeKeep': edge_keep
            }
            
            # Use composite score to select best result
            composite_score = (
                composite_score_weights['ARI'] * metrics['ARI'] +
                composite_score_weights['NMI'] * metrics['NMI'] +
                composite_score_weights['modularity_original_graph'] * metrics['modularity_original_graph']
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_result = {
                    'params': params,
                    'metrics': metrics,
                    'partition': partition,
                    'G_emb': G_emb
                }
       
        except Exception as e:
            print(f"Parameter combination failed: {str(e)}")
            continue
    
    # Output best result
    print("\nBest Result:")
    print(f"Parameters: n_components={best_result['params']['n_components']}, resolution={best_result['params']['resolution']}")
    print("Evaluation Metrics:")
    for metric, value in best_result['metrics'].items():
        print(f"  {metric}: {value:.6f}")
