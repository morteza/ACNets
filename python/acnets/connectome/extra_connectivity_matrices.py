import numpy as np
from sklearn.covariance import LedoitWolf
from nilearn import connectome


class ExtraConnectivityMeasure(connectome.ConnectivityMeasure):

  def __init__(self, cov_estimator=LedoitWolf(store_precision=False),
               kind='covariance', vectorize=False, discard_diagonal=False):

    self.kind = kind
    self.cov_estimator = cov_estimator
    self.vectorize = vectorize
    self.discard_diagonal = discard_diagonal

    if kind.lower() == 'transfer_entropy':
      if vectorize:
        raise ValueError('`vectorize=True` cannot be used with transfer entropy.')
      if discard_diagonal:
        raise ValueError('Discard diagonal cannot be used with non-vectorized connectivities.')
    else:
      super().__init__(cov_estimator, kind, vectorize, discard_diagonal)

  def fit(self, X, y=None):
    if self.kind.lower() == 'transfer_entropy':
      return self
    else:
      return super().fit(X, y)

  def transform(self, X, confounds=None):
    if self.kind.lower() == 'transfer_entropy':
      return self._transfer_entropy_matrix(X)
    else:
      return super().transform(X, confounds)

  def fit_transform(self, X, y=None, confounds=None):
    if self.kind.lower() == 'transfer_entropy':
      return self._transfer_entropy_matrix(X)
    else:
      return super().fit_transform(X, y, confounds)

  def _transfer_entropy_matrix(self, X) -> np.ndarray:
    """Calculate transfer entropy between all pairs of nodes in X.

    Parameters
    ----------
    X : numpy.ndarray
        time series of shape (n_subjects, n_regions, n_timepoints)

    Returns
    -------
    np.ndarray
        Adjacency matrix of size (n_subjects, n_regions, n_regions)
    """

    import sys
    import pyinform

    X_te = []
    for X_sub in X:

      # X is in scikit format, replace region and time axes
      X_sub = X_sub.transpose(1, 0)

      X_sub -= X_sub.mean(axis=1, keepdims=True)
      X_sub /= X_sub.std(axis=1, keepdims=True) + sys.float_info.epsilon
      X_sub, *_ = pyinform.utils.bin_series(X_sub, b=101)

      subj_te = np.zeros((X_sub.shape[0], X_sub.shape[0]))

      for i, source in enumerate(X_sub):
        for j, target in enumerate(X_sub):
          subj_te[i, j] = pyinform.transfer_entropy(source, target, k=1)

      X_te.append(subj_te)

    X_te = np.stack(X_te)

    return X_te
