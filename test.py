import numpy as np
from barneshut import tsne_barnes_hut  
from sklearn.datasets import load_digits

def test_output_shape():
    X, _ = load_digits(return_X_y=True)
    X = X[:100]
    Y = tsne_barnes_hut(X, n_iter=10)
    assert Y.shape == (100, 2)

def test_no_nans():
    X, _ = load_digits(return_X_y=True)
    X = X[:100]
    Y = tsne_barnes_hut(X, n_iter=10)
    assert not np.isnan(Y).any()

def test_no_infs():
    X, _ = load_digits(return_X_y=True)
    X = X[:100]
    Y = tsne_barnes_hut(X, n_iter=10)
    assert np.all(np.isfinite(Y))
