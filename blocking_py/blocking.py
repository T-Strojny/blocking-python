import numpy as np
import pandas as pd
from typing import List, Union, Optional
from scipy import sparse
import os
import platform
import text2vec
from tokenizers import tokenize_character_shingles
import networkx as nx
from collections import OrderedDict

from blocking_py import method_nnd, method_annoy, method_hnsw, method_mlpack



def blocking(x,
             y: Optional[Union[np.ndarray, List]] = None,
             deduplication: bool = True,
             on: Optional[List[str]] = None,
             on_blocking: Optional[List[str]] = None,
             ann: str = None,
             distance = None,
             ann_write: Optional[str] = None,
             ann_colnames: Optional[List[str]] = None,
             true_blocks: Optional[Union[np.ndarray, List]] = None,
             verbose: int = 0,
             graph: bool = False,
             seed: int = 2023,
             n_threads: int = 1,
             control_txt: dict = None,
             control_ann: dict = None
             ):
    
    if ann not in ['nnd', 'hnsw', 'annoy', 'lsh', 'kd']: 
        raise ValueError("ANN should be 'nnd, hnsw, annoy, lsh, kd'")

    if distance is None:
        distance = {  
            "nnd": "cosine",
            "hnsw": "cosine",
            "annoy": "angular",
            "lsh": None,
            "kd": None
        }.get(ann)

    if not (isinstance(x, (str, np.ndarray)) or sparse.issparse(x)):
        raise ValueError("Only character, dense or sparse (csr_matrix) matrix x is supported")
    
    if ann_write is not None:
        if not os.path.exists(ann_write):
            raise ValueError("Path provided in the 'ann_write' is incorrect")
        
    if ann == "hnsw":
        if distance not in ["l2", "euclidean", "cosine", "ip"]:
            raise ValueError("Distance for HNSW should be 'l2, euclidean, cosine, ip'")
        
    if ann == "annoy":
        if distance not in ["euclidean", "manhattan", "hamming", "angular"]:
            raise ValueError("Distance for Annoy should be 'euclidean, manhattan, hamming, angular'")
        
    if y is not None:
        deduplication = False
        y_default = False
        k = 1
    else :
        y_default = y
        y = x
        k = 2

    if true_blocks is not None:
        if not isinstance(true_blocks, pd.DataFrame):
            raise ValueError("'true_blocks' should be a DataFrame")
        
        if not deduplication:
            if true_blocks.shape[1] != 3 or not all(true_blocks.columns == ["x", "y", "block"]):
                raise ValueError("'true blocks' should be a DataFrame with columns: x, y, block")

        if deduplication:
            if true_blocks.shape[1] != 2 or not all(true_blocks.columns == ["x", "block"]):
                raise ValueError("'true blocks' should be a DataFrame with columns: x, block")
            
    if sparse.issparse(x):
        x_dtm = x
        y_dtm = y
    else:
        
        if verbose in [1,2]:
            print("===== creating tokens =====\n")

        if platform.system() in ["Linux", "Darwin"]:  
            x_tokens = text2vec.itoken_parallel(
                iterable=x,
                tokenizer=lambda x: tokenize_character_shingles(
                    x,
                    n=control_txt['n_shingles'],
                    lowercase=control_txt['lowercase'],
                    strip_non_alphanum=control_txt['strip_non_alphanum']
                ),
                n_chunks=control_txt['n_chunks'],
                progressbar=verbose
            )
        else:
            x_tokens = text2vec.itoken(
                iterable = x,
                tokenizer = lambda x: tokenize_character_shingles(x,
                    n=control_txt['n_shingles'],
                    lowercase=control_txt['lowercase'],
                    strip_non_alphanum=control_txt['strip_non_alphanum']),
                n_chunks = control_txt['n_chunks'],
                progressbar = verbose
            )

        x_voc = text2vec.create_vocabulary(x_tokens)
        x_vec = text2vec.vocav_vectorizer(x_voc)
        x_dtm = text2vec.create_dtm(x_tokens, x_vec)

        if y is None:
            y_dtm = x_dtm
        else:
            if platform.system() in ["Linux", "Darwin"]:
                y_tokens = text2vec.itoken_parallel(
                    iterable = y,
                    tokenizer=lambda x: tokenize_character_shingles(
                        x,
                        n=control_txt['n_shingles'],
                        lowercase=control_txt['lowercase'],
                        strip_non_alphanum=control_txt['strip_non_alphanum']
                    ),
                    n_chunks = control_txt['n_chunks'],
                    progressbar = verbose
                ) 
            else:
                y_tokens = text2vec.itoken(
                    iterable = y,
                    tokenizer = lambda x: tokenize_character_shingles(x,
                        n=control_txt['n_shingles'],
                        lowercase=control_txt['lowercase'],
                        strip_non_alphanum=control_txt['strip_non_alphanum']
                        ),
                    n_chunks = control_txt['n_chunks'],
                    progressbar = verbose
                )
            y_voc = text2vec.create_vocabulary(y_tokens)
            y_vec = text2vec.vocav_vectorizer(y_voc)
            y_dtm = text2vec.create_dtm(y_tokens, y_vec)
    
    colnames_xy = np.intersect1d(x_dtm.columns, y_dtm.columns)

    if verbose in [1, 2]:
        print(f"===== starting search ({ann}, x, y: {x_dtm.shape[0]}, {y_dtm.shape[0]}, t: {len(colnames_xy)}) =====")


    if ann == 'nnd':
        x_df = method_nnd(x = x_dtm[:, colnames_xy],
                          y = y_dtm[:, colnames_xy],
                          k = k,
                          distance = distance,
                          deduplication = deduplication,
                          verbose = True if verbose == 2 else False,
                          n_threads = n_threads,
                          control = control_ann)
    elif ann == 'hnsw':
        x_df = method_hnsw(x = x_dtm[:, colnames_xy],
                            y = y_dtm[:, colnames_xy],
                            k = k,
                            distance = distance,
                            verbose = True if verbose == 2 else False,
                            n_threads = n_threads,
                            path = ann_write,
                            control = control_ann)
    elif ann == 'lsh':
        x_df = method_mlpack(x = x_dtm[:, colnames_xy],
                            y = y_dtm[:, colnames_xy],
                            algo = 'lsh',
                            k = k,
                            verbose = True if verbose == 2 else False,
                            seed = seed,
                            path = ann_write,
                            control = control_ann)
    elif ann == 'kd':
        x_df = method_mlpack(x = x_dtm[:, colnames_xy],
                            y = y_dtm[:, colnames_xy],
                            algo = 'kd',
                            k = k,
                            verbose = True if verbose == 2 else False,
                            seed = seed,
                            path = ann_write,
                            control = control_ann)
    elif ann == 'annoy':
        x_df = method_annoy(x = x_dtm[:, colnames_xy],
                            y = y_dtm[:, colnames_xy],
                            k = k,
                            distance = distance,
                            verbose = True if verbose == 2 else False,
                            path = ann_write,
                            seed = seed, 
                            control = control_ann)
    
    if verbose in [1,2]:
        print("===== creating graph =====\n")

    if deduplication:
        x_df = x_df[x_df['y'] > x_df['x']]

        x_df['query_g'] = 'q' + x_df['y'].astype(str)
        x_df['index_g'] = 'q' + x_df['x'].astype(str)
    else:
        x_df['query_g'] = 'q' + x_df['y'].astype(str)
        x_df['index_g'] = 'i' + x_df['x'].astype(str)
    
    ### IGRAPH PART IN R
    x_gr = nx.from_pandas_edgelist(x_df, source='query_g', target='index_g', create_using=nx.Graph())
    components = nx.connected_components(x_gr)
    x_block = {}
    for component_id, component in enumerate(components):
        for node in component:
            x_block[node] = component_id

    unique_query_g = x_df['query_g'].unique()
    unique_index_g = x_df['index_g'].unique()
    combined_keys = list(unique_query_g) + [node for node in unique_index_g if node not in unique_query_g]

    sorted_dict = OrderedDict()
    for key in combined_keys:
        if key in x_block:
            sorted_dict[key] = x_block[key]

    x_df['block'] = x_df['query_g'].apply(lambda x: x_block[x] if x in x_block else None)
    ###

    if true_blocks is not None:
        if not deduplication:
            pairs_to_eval = x_df[x_df['y'].isin(true_blocks['y'])][['x','y','block']]
            pairs_to_eval = pairs_to_eval.merge(true_blocks[['x','y']],
                                                on=['x','y'],
                                                how='left',
                                                indicator='both')
            pairs_to_eval['both'] = np.where(pairs_to_eval['both'] == 'both',0,-1)

            true_blocks = true_blocks.merge(pairs_to_eval[['x', 'y']], 
                                            on=['x', 'y'], 
                                            how='left', 
                                            indicator='both')
            true_blocks['both'] = np.where(true_blocks['both'] == 'both', 0, 1)
            true_blocks['block'] = true_blocks['block'] + pairs_to_eval['block'].max()

            to_concat = true_blocks[true_blocks['both'] == 1][['x', 'y', 'block', 'both']]
            pairs_to_eval = pd.concat([pairs_to_eval, to_concat], ignore_index=True)
            pairs_to_eval['row_id'] = range(len(pairs_to_eval))
            pairs_to_eval['x2'] = pairs_to_eval['x'] + pairs_to_eval['y'].max()

            pairs_to_eval_long = pd.melt(pairs_to_eval[['y', 'x2', 'row_id', 'block', 'both']],
                                        id_vars=['row_id', 'block', 'both'],
                                        )
            pairs_to_eval_long = pairs_to_eval_long[pairs_to_eval_long['both'] == 0]
            pairs_to_eval_long['block_id'] = pairs_to_eval_long.groupby('block').ngroup()
            pairs_to_eval_long['true_id'] = pairs_to_eval_long['block_id']

            block_id_max = pairs_to_eval_long['block_id'].max(skipna=True)
            pairs_to_eval_long.loc[pairs_to_eval_long['both'] == -1, 'block_id'] = block_id_max + pairs_to_eval_long.groupby('row_id').ngroup() + 1 
            block_id_max = pairs_to_eval_long['block_id'].max(skipna=True)
            # recreating R's rleid function
            pairs_to_eval_long['rleid'] = (pairs_to_eval_long['row_id'] != pairs_to_eval_long['row_id'].shift(1)).cumsum()
            pairs_to_eval_long.loc[(pairs_to_eval_long['both'] == 1) & (pairs_to_eval_long['block_id'].isna()), 'block_id'] = block_id_max + pairs_to_eval_long['rleid']

            true_id_max = pairs_to_eval_long['true_id'].max(skipna=True)
            pairs_to_eval_long.loc[pairs_to_eval_long['both'] == 1, 'true_id'] = true_id_max + pairs_to_eval_long.groupby('row_id').ngroup() + 1
            true_id_max = pairs_to_eval_long['treu_id'].max(skipna=True)
            # recreating R's rleid function again
            pairs_to_eval_long['rleid'] = (pairs_to_eval_long['row_id'] != pairs_to_eval_long['row_id'].shift(1)).cumsum()
            pairs_to_eval_long.loc[(pairs_to_eval_long['both'] == -1) & (pairs_to_eval_long['true_id'].isna()), 'true_id'] = true_id_max + pairs_to_eval_long['rleid']

            pairs_to_eval_long.drop('rleid', inplace=True)

        else:
            #pairs_to_eval_long <- melt(x_df[, .(x,y,block)], id.vars = c("block"))
            #pairs_to_eval_long <- unique(pairs_to_eval_long[, .(block_id=block, x=value)])
            #pairs_to_eval_long[true_blocks, on = "x", true_id := i.block]
