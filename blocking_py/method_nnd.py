import pynndescent
import pandas as pd
import numpy as np

def method_nnd(x, y, k, distance, deduplication, verbose, n_threads, control):

    if k > x.shape[0]:
        raise ValueError("k is bigger than the size of x")

    l_ind = pynndescent.NNDescent(data = x,
                                  n_neighbors = x.shape[0] if control['k_search'] > x.shape[0] else control['k_search'],
                                  metric = distance,
                                  metric_kwds = control['nnd']['metric_kwds'],
                                  verbose = verbose,
                                  n_jobs = n_threads,
                                  tree_init = control['nnd']['tree_init'],
                                  n_trees = control['nnd']['n_trees'],
                                  leaf_size = control['nnd']['leaf_size'],
                                  #This argument is in documentation but does not exist.
                                  #max_rptree_depth = control['nnd']['max_tree_depth'],
                                  n_iters = control['nnd']['n_iters'],
                                  delta = control['nnd']['delta'],
                                  max_candidates = control['nnd']['max_candidates'],
                                  low_memory = control['nnd']['low_memory'],
                                  n_search_trees = control['nnd']['n_search_trees'],
                                  pruning_degree_multiplier = control['nnd']['pruning_degree_multiplier'],
                                  diversify_prob = control['nnd']['diversify_prob'],
                                  init_dist = control['nnd']['init_dist'],
                                  init_graph = control['nnd']['init_graph'],
                                  random_state = control['nnd']['random_state'],
                                  compressed = control['nnd']['compressed'],
                                  parallel_batch_queries = control['nnd']['parallel_batch_queries']
                                    )
    
    l_1nn = l_ind.query(query_data = y,
                        k = x.shape[0] if control['k_search'] > x.shape[0] else control['k_search'],
                        epsilon = control['nnd']['epsilon']
                        )
    
    l_df = pd.DataFrame({
            'y': np.arange(0, y.shape[0]),
            'x': l_1nn[0][:, k-1], 
            'dist': l_1nn[1][:, k-1]
        })
    
    return l_df


### EXAMPLE USAGE

# x = np.random.rand(100, 10)  
# y = np.random.rand(5, 10)    


# control = {
#     'k_search': 10,  
#     'nnd': {
#         'metric_kwds': {},  
#         'tree_init': True,
#         'n_trees': 10,
#         'leaf_size': 30,
#         'max_tree_depth': 20,
#         'n_iters': 10,
#         'delta': 0.001,
#         'max_candidates': 60,
#         'low_memory': False,
#         'n_search_trees': 1,
#         'pruning_degree_multiplier': 1.5,
#         'diversify_prob': 0.1,
#         'init_dist': 'euclidean',
#         'init_graph': None,
#         'random_state': 42,
#         'compressed': False,
#         'parallel_batch_queries': True,
#         'epsilon': 0.1
#     }
# }

# 
# result_df = method_nnd(x, y, k=3, distance='euclidean', deduplication=False, verbose=True, n_threads=2, control=control)

#
# print(result_df.head())
