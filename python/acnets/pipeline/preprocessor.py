# to preprocess the data

from sklearn.base import TransformerMixin, BaseEstimator


class Preprocessor(TransformerMixin, BaseEstimator):
  def __init__(self) -> None:
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    raise NotImplementedError()

  def transform(self, X):
    raise NotImplementedError()
