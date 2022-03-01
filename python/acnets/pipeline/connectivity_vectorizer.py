# to extract connectivity from time series

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class ConnectivityVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self, discard_diagonal=False, discard_tril=True) -> None:
    self.discard_tril = discard_tril
    self.k = 1 if discard_diagonal else 0
    super().__init__()

  def fit(self, X, y=None, **fit_params):  # noqa: N803
    return self

  def transform(self, X):  # noqa: N803
    if X.ndim != 2:
      raise ValueError('Input must be a 2D array.')

    if X.shape[0] != X.shape[1]:
      raise ValueError('Input must be a square array.')

    if self.discard_tril:
      X_vec = X[np.triu_indices(X.shape[0], k=self.k)]
    else:
      X_vec = X.flatten()

    return X_vec
