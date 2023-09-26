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

    return self

  def transform(self, dataset, y=None):  # noqa: N803
    self.dataset_ = dataset

    self.node_type = self.dataset_['timeseries'].dims[-1]

    self.dataset_[self.node_type + '_src'] = self.dataset_[self.node_type].values
    self.dataset_[self.node_type + '_dst'] = self.dataset_[self.node_type].values

    timeseries = self.dataset_['timeseries'].values
    self.conn_estimator.fit(timeseries, y)

    timeseries = self.dataset_['timeseries'].values
    conn = self.conn_estimator.transform(timeseries)

    self.dataset_['connectivity'] = (
        ['subject', self.node_type + '_src', self.node_type + '_dst'],
        conn)

    return self.dataset_

  def get_feature_names_out(self, input_features=None, sep=' \N{left right arrow} '):
    return self.dataset_.coords[self.node_type].values.tolist()
