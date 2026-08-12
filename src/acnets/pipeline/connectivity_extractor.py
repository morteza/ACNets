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
    self.node_type = dataset['timeseries'].dims[-1]
    self.feature_names_ = dataset.coords[self.node_type].values.tolist()
    return self

  def transform(self, dataset, y=None):  # noqa: N803
    new_dataset = dataset.copy()

    self.node_type = new_dataset['timeseries'].dims[-1]

    new_dataset[self.node_type + '_src'] = new_dataset[self.node_type].values
    new_dataset[self.node_type + '_dst'] = new_dataset[self.node_type].values

    timeseries = new_dataset['timeseries'].values
    self.conn_estimator.fit(timeseries, y)

    timeseries = new_dataset['timeseries'].values
    conn = self.conn_estimator.transform(timeseries)

    new_dataset['connectivity'] = (
        ['subject', self.node_type + '_src', self.node_type + '_dst'],
        conn)

    # record which statistic produced this matrix. A connectivity dataset that
    # has lost track of its own kind cannot be told apart from another once the
    # two are side by side, which is the normal situation in the multiverse.
    new_dataset['connectivity'].attrs['kind'] = self.kind
    new_dataset.attrs['connectivity_kind'] = self.kind

    return new_dataset

  def get_feature_names_out(self, input_features=None, sep=' \N{left right arrow} '):
    return self.feature_names_
