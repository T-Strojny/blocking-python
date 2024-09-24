import pytest
import numpy as np
import pandas as pd
from blocking_py.method_nnd import method_nnd

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.random.rand(5, 10)
    return x, y

@pytest.fixture
def control():
    return {
        'k_search': 10,
        'nnd': {
            'metric_kwds': {},
            'tree_init': True,
            'n_trees': 10,
            'leaf_size': 30,
            'n_iters': 10,
            'delta': 0.001,
            'max_candidates': 60,
            'low_memory': False,
            'n_search_trees': 1,
            'pruning_degree_multiplier': 1.5,
            'diversify_prob': 0.1,
            'init_dist': 'euclidean',
            'init_graph': None,
            'random_state': 42,
            'compressed': False,
            'parallel_batch_queries': True,
            'epsilon': 0.1
        }
    }

def test_method_nnd_output_shape(sample_data, control):
    x, y = sample_data
    result = method_nnd(x, y, k=3, distance='euclidean', deduplication=False, verbose=False, n_threads=1, control=control)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (y.shape[0], 3)
    assert set(result.columns) == {'y', 'x', 'dist'}

def test_method_nnd_output_values(sample_data, control):
    x, y = sample_data
    result = method_nnd(x, y, k=3, distance='euclidean', deduplication=False, verbose=False, n_threads=1, control=control)
    assert (result['y'] == np.arange(y.shape[0])).all()
    assert (result['x'] >= 0).all() and (result['x'] < x.shape[0]).all()
    assert (result['dist'] >= 0).all()

def test_method_nnd_k_parameter(sample_data, control):
    x, y = sample_data
    for k in [1, 3, 5]:
        result = method_nnd(x, y, k=k, distance='euclidean', deduplication=False, verbose=False, n_threads=1, control=control)
        assert result.shape == (y.shape[0], 3)

def test_method_nnd_distance_parameter(sample_data, control):
    x, y = sample_data
    for distance in ['euclidean', 'manhattan', 'cosine']:
        result = method_nnd(x, y, k=3, distance=distance, deduplication=False, verbose=False, n_threads=1, control=control)
        assert result.shape == (y.shape[0], 3)

def test_method_nnd_n_threads(sample_data, control):
    x, y = sample_data
    for n_threads in [1, 2, 4]:
        result = method_nnd(x, y, k=3, distance='euclidean', deduplication=False, verbose=False, n_threads=n_threads, control=control)
        assert result.shape == (y.shape[0], 3)

def test_method_nnd_k_search(sample_data, control):
    x, y = sample_data
    for k_search in [5, 10, 20]:
        control['k_search'] = k_search
        result = method_nnd(x, y, k=3, distance='euclidean', deduplication=False, verbose=False, n_threads=1, control=control)
        assert result.shape == (y.shape[0], 3)

if __name__ == "__main__":
    pytest.main()