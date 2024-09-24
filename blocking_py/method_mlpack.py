from mlpack import lsh, knn
import pandas as pd
import numpy as np
from scipy.sparse import issparse


def method_mlpack(x, y, algo, k, verbose, seed, path, control):
    if algo not in ['lsh', 'kd']:
        raise ValueError("Incorrect algorithm specified. Available options are 'lsh' or 'kd'")

    if issparse(x):
        x = x.toarray()
    if issparse(y):
        y = y.toarray()

    if algo == 'lsh':
        result = lsh(k = x.shape[0] if control['k_search'] > x.shape[0] else control['k_search'],
                   query = y,
                   reference = x,
                   verbose = verbose,
                   seed = seed,
                   bucket_size = control['lsh']['bucket_size'],
                   hash_width = control['lsh']['hash_width'],
                   num_probes = control['lsh']['num_probes'],
                   projections = control['lsh']['projections'],
                   tables = control['lsh']['tables'])
    else:
        result = knn(k = x.shape[0] if control['k_search'] > x.shape[0] else control['k_search'],
                   query = y,
                   reference = x,
                   verbose = verbose,
                   seed = seed,
                   algorithm = control['kd']['algorithm'],
                   leaf_size = control['kd']['leaf_size'],
                   tree_type = control['kd']['tree_type'],
                   epsilon = control['kd']['epsilon'],
                   rho = control['kd']['rho'],
                   tau = control['kd']['tau'],
                   random_basis = control['kd']['random_basis'])
        
    l_df = pd.DataFrame({
            'y': np.arange(0,y.shape[0]),
            'x': result['neighbors'][:, k-1], 
            'dist': result['distances'][:, k-1]
        })
    
    return l_df

### EXAMPLE USAGE

# from scipy.sparse import csr_matrix
# reference_data = np.random.rand(10, 5)  
# query_data = np.random.rand(5, 5)       
# control = {
#     'k_search': 3,  
#     'lsh': {
#         'bucket_size': 20,
#         'hash_width': 0.5,
#         'num_probes': 5,
#         'projections': 50,
#         'tables': 10
#     },
#     'kd': {
#         'algorithm': 'dual_tree',
#         'leaf_size': 20,
#         'tree_type': 'kd',
#         'epsilon': 0.01,
#         'rho': 0.1,
#         'tau': 0.5,
#         'random_basis': False
#     }
# }
# result_lsh = method_mlpack(
#     x=reference_data,
#     y=query_data,
#     algo="lsh",
#     k=3, 
#     verbose=True,
#     seed=12345,
#     path=None,
#     control=control
# )

# result_kd = method_mlpack(
#     x=reference_data,
#     y=query_data,
#     algo="kd",
#     k=3, 
#     verbose=True,
#     seed=12345,
#     path=None,
#     control=control
# )

# result_kd.head()
