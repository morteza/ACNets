# to extract connectivity from time series

from sklearn.base import TransformerMixin, BaseEstimator
from ..connectome import ExtraConnectivityMeasure


class ConnectivityExtractor(TransformerMixin, BaseEstimator):
  def __init__(self, kind='correlation') -> None:
    self.conn_estimator = ExtraConnectivityMeasure(kind=kind, vectorize=False)
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    return self.conn_estimator.fit(X, y, **fit_params)

  def transform(self, X):
    return self.conn_estimator.transform(X)
