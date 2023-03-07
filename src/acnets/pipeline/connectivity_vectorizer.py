# to extract connectivity from time series

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import xarray as xr
import pandas as pd


class ConnectivityVectorizer(TransformerMixin, BaseEstimator):
  def __init__(self,
               discard_tril=True,
               discard_diagonal=False,
               only_diagonal=False):
    self.discard_tril = discard_tril
    self.discard_diagonal = discard_diagonal
    self.only_diagonal = only_diagonal
    self.dicard_diagonal = discard_diagonal
    self.k = 1 if discard_diagonal else 0

    if only_diagonal and discard_diagonal:
      raise ValueError('Cannot keep diagonal values while discarding them.')

    super().__init__()

  def fit(self, X, y=None, **fit_params):  # noqa: N803
    return self

  def transform(self, X):  # noqa: N803

    if isinstance(X, xr.DataArray):
      X = X.values

    if X.ndim not in [2, 3]:
      raise ValueError('Input must be a 2D array of shape (regions, regions)'
                       ' or 3D array of shape (subjects, regions, regions).')

    if X.ndim == 2:
      X_vec = self._vectorize_single_subject(X)
    elif X.ndim == 3:
      X_vec = np.array([self._vectorize_single_subject(X_subj) for X_subj in X])

    return X_vec

  def _vectorize_single_subject(self, X_subj):  # noqa: N803
    if X_subj.shape[0] != X_subj.shape[1]:
      raise ValueError('Connectivity matrix must be a square array.')

    if self.only_diagonal:
      X_vec = X_subj.diagonal()
    elif self.discard_tril:
      X_vec = X_subj[np.triu_indices(X_subj.shape[0], k=self.k)]
    else:
      X_vec = X_subj.flatten()

    return X_vec

  def get_feature_names_out(self, input_features):
      sep = ' \N{left right arrow} '
      feature_names = input_features.stack().to_frame().apply(lambda x:
          sep.join(x.name) if x.name[0] != x.name[1] else x.name[0],
          axis=1).unstack()
      feature_names = self.transform(feature_names.values)
      return feature_names

  # DEBUG lbls = get_feature_labels('gordon2014_2mm', 'tangent', True)