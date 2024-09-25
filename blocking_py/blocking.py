import numpy as np
import pandas as pd
from typing import List, Union, Optional
from scipy import sparse
import os


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
            
    # To do : reszta