# to combine region-level connectivities into confirmatory network-level factors

from sklearn.base import TransformerMixin, BaseEstimator


class ConfirmatoryFactorAnalyser(TransformerMixin, BaseEstimator):
  def __init__(self) -> None:
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    raise NotImplementedError()

  def transform(self, X):
    raise NotImplementedError()
