import pytest
import numpy as np
from src.acnets.pipeline import ConnectivityVectorizer
from src.acnets.pipeline import ConnectivityExtractor
from src.acnets.pipeline import Parcellation


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


@pytest.mark.parametrize('atlas_name,kind', [
    ('dosenbach2007', 'tangent'),
    ('dosenbach2010', 'correlation'),
    ('difumo_64_2mm', 'chatterjee'),
    ('difumo_64_2mm', 'tangent'),
])
def test_triu_connectivity_vectorizer(atlas_name, kind):

  dataset = Parcellation(atlas_name=atlas_name).fit().dataset
  timeseries = dataset['timeseries'].values

  extractor = ConnectivityExtractor(kind=kind, verbose=1)
  all_conns = extractor.fit_transform(timeseries)

  for conn in all_conns:
    vectorizer = ConnectivityVectorizer(discard_tril=True,
                                        discard_diagonal=False,
                                        only_diagonal=False)
    conn_vec = vectorizer.fit_transform(conn)

    n_dim = conn.shape[0]

    assert conn_vec.shape == (n_dim * (n_dim + 1) / 2,)


@pytest.mark.parametrize('atlas_name,kind', [
    ('dosenbach2007', 'tangent'),
    ('dosenbach2010', 'tangent'),
    ('difumo_64_2mm', 'tangent'),
])
def test_diagonal_connectivity_vectorizer(atlas_name, kind):

  dataset = Parcellation(atlas_name=atlas_name).fit().dataset
  timeseries = dataset['timeseries'].values

  extractor = ConnectivityExtractor(kind=kind, verbose=1)
  all_conns = extractor.fit_transform(timeseries)

  for conn in all_conns:
    vectorizer = ConnectivityVectorizer(only_diagonal=True)
    conn_vec = vectorizer.fit_transform(conn)

    n_dim = conn.shape[0]

    assert conn_vec.shape == (n_dim,)
