import pytest
import numpy as np
from sqlalchemy import false
from python.acnets.pipeline.connectivity_vectorizer import ConnectivityVectorizer
import sys


@pytest.mark.parametrize('n_dim', [1, 2, 3, 10, 100])
def test_triu_vectorizer(n_dim):

  X = np.random.rand(n_dim, n_dim)
  flatten_dim = n_dim * (n_dim + 1) // 2

  vectorizer = ConnectivityVectorizer(discard_tril=True)
  X_vec = vectorizer.fit_transform(X)

  assert X_vec.shape == (flatten_dim,) or (X_vec.shape == flatten_dim)


@pytest.mark.parametrize('n_dim', [1, 2, 3, 100])
def test_full_vectorizer(n_dim):

  X = np.random.rand(n_dim, n_dim)

  vectorizer = ConnectivityVectorizer(discard_tril=False)
  X_vec = vectorizer.fit_transform(X)

  assert X_vec.shape == (n_dim**2,)
