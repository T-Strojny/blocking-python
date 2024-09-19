from pathlib import Path
import numpy as np
from annoy import AnnoyIndex
import tempfile
import pandas as pd


def method_annoy(x, y, k, distance, verbose, path, seed, control):
    if k > x.shape[0]:
        raise ValueError("k is bigger than the size of x")
    
    ncols = x.shape[1]

    if distance == "euclidean":
        l_ind = AnnoyIndex(ncols, 'euclidean')
    elif distance in ["manhattan", "manhatan"]:  
        l_ind = AnnoyIndex(ncols, 'manhattan')
    elif distance == "hamming":
        l_ind = AnnoyIndex(ncols, 'hamming')
    elif distance == "angular":
        l_ind = AnnoyIndex(ncols, 'angular')
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    l_ind.set_seed(seed)

    if control['annoy']['build_on_disk']:
        temp_annoy = tempfile.mktemp(prefix="annoy", suffix=".tree")
        print(f'Building index on disk: {temp_annoy}')
        l_ind.on_disk_build(temp_annoy)

    if verbose:
        l_ind.verbose(True)
        print("Building index...")

    
    for i in range(x.shape[0]):
        l_ind.add_item(i, x[i])
    
    l_ind.build(control['annoy']['n_trees'])

    l_ind_nns = np.zeros(y.shape[0], dtype=int)
    l_ind_dist = np.zeros(y.shape[0])

    if verbose:
        print("Querying index...")
    for i in range(y.shape[0]):
        k_search = min(x.shape[0], control['k_search'])
        annoy_res = l_ind.get_nns_by_vector(y[i], k_search, include_distances=True)
        l_ind_nns[i] = annoy_res[0][k-1] 
        l_ind_dist[i] = annoy_res[1][k-1]  

    if path is not None:
        try:
            path = Path(path)
            if verbose:
                print(f"Saving index to {path}")
            
            path.mkdir(parents=True, exist_ok=True)
            
            index_path = path / "index.annoy"
            colnames_path = path / "index-colnames.txt"
            
            l_ind.save(str(index_path))
            
            if index_path.exists():
                file_size = index_path.stat().st_size
                if file_size > 0:
                    print(f"Index saved successfully to {index_path}. File size: {file_size} bytes")
                else:
                    print(f"Warning: Index file {index_path} was created but is empty.")
            else:
                print(f"Error: Failed to create index file {index_path}")

            np.savetxt(colnames_path, np.arange(ncols), fmt='%d')
            
            if verbose:
                print(f"Index saved to: {index_path}")
                print(f"Column names saved to: {colnames_path}")
        except Exception as e:
            print(f"Error saving index or column names: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()

    l_df = {
        'y': np.arange(y.shape[0]),  
        'x': l_ind_nns, 
        'dist': l_ind_dist,
    }

    l_df = pd.DataFrame(l_df)

    if verbose:
        print("Process completed successfully.")
    return l_df

# control = {
#     'annoy': {
#         'build_on_disk': False,
#         'n_trees': 5,
#     },
#     'k_search': 5
# }
#
# x = np.random.rand(100, 2)
# y = np.array([[6.776, 3.456], [1.234, 2.345]])
# path = r"path"

# result = method_annoy(x, y, k=1, distance="euclidean", verbose=True, path=False, seed=42, control=control)
# print(result)