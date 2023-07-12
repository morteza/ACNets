# to extract connectivity from time series

from sklearn.base import TransformerMixin, BaseEstimator
from ..connectome import ExtraConnectivityMeasure


class ConnectivityExtractor(TransformerMixin, BaseEstimator):
  def __init__(self, kind='correlation', verbose=0) -> None:
    self.kind = kind
    self.verbose = verbose
    self.conn_estimator = ExtraConnectivityMeasure(kind=kind, vectorize=False)
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):
    self.dataset_ = dataset
    self.node_type = dataset['timeseries'].dims[-1]

    timeseries = self.dataset_['timeseries'].values
    self.conn_estimator.fit(timeseries, y, **fit_params)

    return self

  def transform(self, dataset):  # noqa: N803
    timeseries = dataset['timeseries'].values
    conn = self.conn_estimator.transform(timeseries)

    self.dataset_['connectivity'] = (
        ['subject', self.node_type, self.node_type],
        conn)

    return self.dataset_
