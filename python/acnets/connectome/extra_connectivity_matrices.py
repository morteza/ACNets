import numpy as np
from sklearn.covariance import LedoitWolf
from nilearn import connectome

from ._transfer_entropy import transfer_entropy
from ._chatterjee import chatterjee_xicoef


class ExtraConnectivityMeasure(connectome.ConnectivityMeasure):

  def __init__(self, cov_estimator=LedoitWolf(store_precision=False),
               kind='covariance', vectorize=False, discard_diagonal=False):

    self.kind = kind
    self.cov_estimator = cov_estimator
    self.vectorize = vectorize
    self.discard_diagonal = discard_diagonal

    if kind == 'transfer_entropy':
      if vectorize:
        raise ValueError('`vectorize=True` cannot be used with transfer entropy.')
      if discard_diagonal:
        raise ValueError('Discard diagonal cannot be used with non-vectorized connectivities.')
    elif kind == 'chatterjee':
      pass
    else:
      super().__init__(cov_estimator, kind, vectorize, discard_diagonal)

  def fit(self, X, y=None):
    if self.kind == 'transfer_entropy':
      return self
    elif self.kind == 'chatterjee':
      return self
    else:
      return super().fit(X, y)

  def transform(self, X, confounds=None):
    if self.kind == 'transfer_entropy':
      return transfer_entropy(X)
    elif self.kind == 'chatterjee':
      return chatterjee_xicoef(X)
    else:
      return super().transform(X, confounds)

  def fit_transform(self, X, y=None, confounds=None):
    if self.kind == 'transfer_entropy':
      return transfer_entropy(X)
    if self.kind == 'chatterjee':
      return chatterjee_xicoef(X)
    else:
      return super().fit_transform(X, y, confounds)
