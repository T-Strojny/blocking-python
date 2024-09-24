import hnswlib
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def method_hnsw(x, y, k, distance, verbose, n_threads, path, control):

    if k > x.shape[0]:
        raise ValueError("k is bigger than the size of x")

    if issparse(x):
        x = x.toarray()
    if issparse(y):
        y = y.toarray()

    # if control['sparse']:
    if distance in ["l2", "euclidean"]:
        index = hnswlib.Index(space='l2', dim = x.shape[1])
    elif distance == "cosine":
        index = hnswlib.Index(space='cosine', dim = x.shape[1])
    elif distance == "ip":
        index = hnswlib.Index(space='ip', dim = x.shape[1])
    else:
        raise ValueError("Unsupported distance metric. Choose from ['l2', 'cosine', 'ip', 'euclidean']")
    
    index.init_index(max_elements=x.shape[0], ef_construction=control['hnsw']['ef_c'], M=control['hnsw']['M'])
    index.set_num_threads(n_threads)
    index.add_items(x)
    index.set_ef(control['hnsw']['ef_s'])

    l_1nn = index.knn_query(y, k=k)

    if path:
        import os
        path_ann = os.path.join(path, "index.hnsw")
        path_ann_cols = os.path.join(path, "index-colnames.txt")
        
        if verbose:
            print("Writing an index to `path`")
        
        index.save_index(path_ann)
        with open(path_ann_cols, 'w') as f:
            f.write('\n'.join(str(col) for col in range(x.shape[1])))

    l_df = pd.DataFrame({
        'y': range(0, y.shape[0]),
        'x': l_1nn[0][:, k-1],
        'dist': l_1nn[1][:, k-1]
    })

    return l_df

### EXAMPLE USAGE

# x = np.random.rand(1000, 50)
# y = np.random.rand(100, 50)
# control = {
#      'hnsw': {'M': 16, 'ef_c': 200, 'ef_s': 100},
#      'k_search': 100
# }

# result = method_hnsw(x, y, k=5, distance='l2', verbose=True, n_threads=4, path=None, control=control)
# print(result)
