import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from blocking_py.method_mlpack import method_mlpack

@pytest.fixture
def control_params():
   
    return {
        'k_search': 3,
        'lsh': {
            'bucket_size': 20,
            'hash_width': 0.5,
            'num_probes': 5,
            'projections': 50,
            'tables': 10
        },
        'kd': {
            'algorithm': 'dual_tree',
            'leaf_size': 20,
            'tree_type': 'kd',
            'epsilon': 0.01,
            'rho': 0.1,
            'tau': 0.5,
            'random_basis': False
        }
    }

@pytest.fixture
def data():
   
    reference_data = np.random.rand(10, 5)
    query_data = np.random.rand(5, 5) 
    return reference_data, query_data

def test_lsh_method(data, control_params):
  
    reference_data, query_data = data
    result = method_mlpack(
        x=reference_data,
        y=query_data,
        algo="lsh",
        k=3,
        verbose=False,
        seed=12345,
        path=None,
        control=control_params
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame."
    assert 'x' in result.columns, "Result should contain 'x' column."
    assert 'dist' in result.columns, "Result should contain 'dist' column."

def test_knn_method(data, control_params):
    
    reference_data, query_data = data
    result = method_mlpack(
        x=reference_data,
        y=query_data,
        algo="kd",
        k=3,
        verbose=False,
        seed=12345,
        path=None,
        control=control_params
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame."
    assert 'x' in result.columns, "Result should contain 'x' column."
    assert 'dist' in result.columns, "Result should contain 'dist' column."

def test_invalid_algorithm(data, control_params):

    reference_data, query_data = data
    
    with pytest.raises(ValueError, match="Incorrect algorithm specified"):
        method_mlpack(
            x=reference_data,
            y=query_data,
            algo="invalid_algo", 
            k=3,
            verbose=False,
            seed=12345,
            path=None,
            control=control_params
        )

def test_sparse_matrix_input(control_params):

    reference_data = csr_matrix(np.random.rand(10, 5))
    query_data = csr_matrix(np.random.rand(5, 5))     

    result = method_mlpack(
        x=reference_data,
        y=query_data,
        algo="lsh",
        k=3,
        verbose=False,
        seed=12345,
        path=None,
        control=control_params
    )
    
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame."
    assert 'x' in result.columns, "Result should contain 'x' column."
    assert 'dist' in result.columns, "Result should contain 'dist' column."

def test_correct_neighbors_and_distances():
    
    reference_data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    
    query_data = np.array([
        [1.1, 2.1],  
        [3.1, 4.1],  
    ])

    control_params = {
        'k_search': 1,  
        'lsh': {
            'bucket_size': 20,
            'hash_width': 0.5,
            'num_probes': 5,
            'projections': 50,
            'tables': 10
        },
        'kd': {
            'algorithm': 'dual_tree',
            'leaf_size': 20,
            'tree_type': 'kd',
            'epsilon': 0.01,
            'rho': 0.1,
            'tau': 0.5,
            'random_basis': False
        }
    }
    
    
    result = method_mlpack(
        x=reference_data,
        y=query_data,
        algo="kd",
        k=1, 
        verbose=False,
        seed=12345,
        path=None,
        control=control_params
    )

    
    expected_neighbors = [0, 2]  
    expected_distances = [np.linalg.norm([1.1 - 1.0, 2.1 - 2.0]),  
                          np.linalg.norm([3.1 - 3.0, 4.1 - 4.0])] 
    assert np.all(result['x'].values == expected_neighbors), "Neighbors do not match the expected values."
     
    np.testing.assert_allclose(result['dist'].values, expected_distances, rtol=1e-5), "Distances do not match expected values."
