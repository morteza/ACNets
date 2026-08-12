from sklearn.covariance import LedoitWolf
from nilearn import connectome

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
      super().__init__(cov_estimator, kind, vectorize, discard_diagonal, standardize='zscore_sample')

  def fit(self, X, y=None):
    if self.kind in ['transfer_entropy', 'chatterjee']:
      return self
    else:
      return super().fit(X, y)

  def transform(self, X, y=None, confounds=None):
    if self.kind == 'transfer_entropy':
      from ._transfer_entropy import transfer_entropy
      return transfer_entropy(X)
    if self.kind == 'chatterjee':
      from ._chatterjee import chatterjee_xicoef
      return chatterjee_xicoef(X)
    else:
      # transform, not fit_transform. The two are interchangeable for the
      # per-subject statistics, but not for `tangent`, which projects each
      # subject onto a group mean covariance learned in fit. Re-fitting here
      # would give the test fold its own reference point, silently making the
      # tangent specifications non-comparable across folds. Callers that want
      # the old behaviour should call fit_transform explicitly.
      return super().transform(X, confounds)
