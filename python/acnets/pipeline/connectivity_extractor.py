# to extract connectivity from time series

from sklearn.base import TransformerMixin, BaseEstimator
from ..connectome import ExtraConnectivityMeasure


class ConnectivityExtractor(TransformerMixin, BaseEstimator):
  def __init__(self, kind='correlation', verbose=0) -> None:
    self.kind = kind
    self.verbose = verbose
    self.conn_estimator = ExtraConnectivityMeasure(kind=kind, vectorize=False)
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    self.conn_estimator.fit(X, y, **fit_params)
    return self

  def transform(self, X):  # noqa: N803
    conn = self.conn_estimator.transform(X)
    return conn
