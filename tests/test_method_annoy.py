import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path
from blocking_py.method_annoy import method_annoy 

@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.random.rand(20, 10)
    return x, y

@pytest.fixture
def control():
    return {
        'annoy': {
            'build_on_disk': False,
            'n_trees': 5,
        },
        'k_search': 10
    }

def test_basic_functionality(sample_data, control):
    x, y = sample_data
    result = method_annoy(x, y, k=5, distance='euclidean', verbose=False, path=None, seed=42, control=control)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (20, 3)
    assert set(result.columns) == {'y', 'x', 'dist'}

def test_different_distance_metrics(sample_data, control):
    x, y = sample_data
    for distance in ['euclidean', 'manhattan', 'hamming', 'angular']:
        result = method_annoy(x, y, k=5, distance=distance, verbose=False, path=None, seed=42, control=control)
        assert isinstance(result, pd.DataFrame)

def test_invalid_distance_metric(sample_data, control):
    x, y = sample_data
    with pytest.raises(ValueError):
        method_annoy(x, y, k=5, distance='invalid_metric', verbose=False, path=None, seed=42, control=control)

# def test_k_larger_than_dataset(sample_data, control):
#     x, y = sample_data
#     result = method_annoy(x, y, k=150, distance='euclidean', verbose=False, path=None, seed=42, control=control)
#     assert result.shape == (20, 3)
#     assert result['x'].max() < 100 

def test_save_index(sample_data, control):
    x, y = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        result = method_annoy(x, y, k=5, distance='euclidean', verbose=True, path=tmpdir, seed=42, control=control)
        assert Path(tmpdir, "index.annoy").exists()
        assert Path(tmpdir, "index-colnames.txt").exists()

def test_verbose_output(sample_data, control, capsys):
    x, y = sample_data
    method_annoy(x, y, k=5, distance='euclidean', verbose=True, path=None, seed=42, control=control)
    captured = capsys.readouterr()
    assert "Building index..." in captured.out
    assert "Querying index..." in captured.out
    assert "Process completed successfully." in captured.out

def test_build_on_disk(sample_data, control):
    x, y = sample_data
    control['annoy']['build_on_disk'] = True
    result = method_annoy(x, y, k=5, distance='euclidean', verbose=True, path=None, seed=42, control=control)
    assert isinstance(result, pd.DataFrame)

def test_seed_reproducibility(sample_data, control):
    x, y = sample_data
    result1 = method_annoy(x, y, k=5, distance='euclidean', verbose=False, path=None, seed=42, control=control)
    result2 = method_annoy(x, y, k=5, distance='euclidean', verbose=False, path=None, seed=42, control=control)
    pd.testing.assert_frame_equal(result1, result2)

def test_different_k_values(sample_data, control):
    x, y = sample_data
    for k in [1, 5, 10]:
        result = method_annoy(x, y, k=k, distance='euclidean', verbose=False, path=None, seed=42, control=control)
        assert result.shape == (20, 3)

if __name__ == "__main__":
    pytest.main()