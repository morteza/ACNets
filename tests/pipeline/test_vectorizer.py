import pytest
import numpy as np
from python.acnets.pipeline.connectivity_vectorizer import ConnectivityVectorizer


@pytest.mark.parametrize('n_dim', [1, 2, 3, 10, 100])
def test_vectorizer(n_dim):
  vectorizer = ConnectivityVectorizer(
      discard_diagonal=False,
      discard_tril=True
  )

  X = np.random.rand(n_dim, n_dim)
  X_vec = vectorizer.fit_transform(X)

  flatten_dim = n_dim * (n_dim + 1) // 2

  assert X_vec.shape == (flatten_dim,)
