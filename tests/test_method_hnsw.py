import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from blocking_py.method_hnsw import method_hnsw

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.random.rand(20, 10)
    return x, y

@pytest.fixture
def control():
    return {
        'sparse': False,
        'hnsw': {'M': 16, 'ef_c': 100, 'ef_s': 50},
        'k_search': 10
    }

def test_basic_functionality(sample_data, control):
    x, y = sample_data
    result = method_hnsw(x, y, k=5, distance='l2', verbose=False, n_threads=1, path=None, control=control)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (20, 3)
    assert set(result.columns) == {'y', 'x', 'dist'}

def test_different_distance_metrics(sample_data, control):
    x, y = sample_data
    for distance in ['l2', 'euclidean', 'cosine', 'ip']:
        result = method_hnsw(x, y, k=5, distance=distance, verbose=False, n_threads=1, path=None, control=control)
        assert isinstance(result, pd.DataFrame)

def test_sparse_input(control):
    from scipy.sparse import csr_matrix
    x = csr_matrix(np.random.rand(100, 10))
    y = csr_matrix(np.random.rand(20, 10))
    control['sparse'] = True
    
    result = method_hnsw(x, y, k=5, distance='l2', verbose=False, n_threads=1, path=None, control=control)
    assert isinstance(result, pd.DataFrame)

def test_invalid_distance_metric(sample_data, control):
    x, y = sample_data
    with pytest.raises(ValueError):
        method_hnsw(x, y, k=5, distance='invalid_metric', verbose=False, n_threads=1, path=None, control=control)

# def test_k_larger_than_dataset(sample_data, control):
#     x, y = sample_data
#     result = method_hnsw(x, y, k=150, distance='l2', verbose=False, n_threads=1, path=None, control=control)
#     assert result.shape == (20, 3)
#     assert result['x'].max() < 100  

def test_save_index(sample_data, control):
    x, y = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        result = method_hnsw(x, y, k=5, distance='l2', verbose=True, n_threads=1, path=tmpdir, control=control)
        assert os.path.exists(os.path.join(tmpdir, "index.hnsw"))
        assert os.path.exists(os.path.join(tmpdir, "index-colnames.txt"))

# def test_multithreading(sample_data, control):
#     x, y = sample_data
#     single_thread_time = timeit.timeit(lambda: method_hnsw(x, y, k=5, distance='l2', verbose=False, n_threads=1, path=None, control=control), number=1)
#     multi_thread_time = timeit.timeit(lambda: method_hnsw(x, y, k=5, distance='l2', verbose=False, n_threads=4, path=None, control=control), number=1)
    
#     assert multi_thread_time < single_thread_time

def test_verbose_output(sample_data, control, capsys):
    x, y = sample_data
    method_hnsw(x, y, k=5, distance='l2', verbose=True, n_threads=1, path=None, control=control)
    captured = capsys.readouterr()
    assert "Writing an index to `path`" not in captured.out

    with tempfile.TemporaryDirectory() as tmpdir:
        method_hnsw(x, y, k=5, distance='l2', verbose=True, n_threads=1, path=tmpdir, control=control)
        captured = capsys.readouterr()
        assert "Writing an index to `path`" in captured.out

if __name__ == "__main__":
    pytest.main()