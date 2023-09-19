from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from typing import Callable, Literal


class ConnectivityAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level connectivity into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal[None, 'network', 'random_network'] = None,
               reduce_fn: Callable = np.mean,
               conn_transform_fn: Callable = np.fabs
               ) -> None:

    self.strategy = strategy

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    if callable(conn_transform_fn):
      self.conn_transform_fn = conn_transform_fn
    else:
      raise ValueError(f'Transformation method {conn_transform_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, dataset):
    node_type = dataset['timeseries'].dims[-1]

    if self.strategy is None:
      return dataset

    # we need region-level connectivity matrices to aggregate
    if node_type != 'region':
      raise ValueError(f'Time-series are already aggregated. '
                       f'Connectivity aggregation strategy `{self.strategy}` is not supported.')

    if self.strategy == 'random_network':
      dataset['network'] = (['region'], np.random.permutation(dataset['network']))

    dataset = dataset.assign_coords(network_src=('region_src', dataset['network'].values))
    dataset = dataset.assign_coords(network_dst=('region_dst', dataset['network'].values))

    # defaults to take absolute value of connectivity matrices
    dataset['connectivity'] = self.conn_transform_fn(dataset['connectivity'])

    dataset['connectivity'] = (
        dataset['connectivity']
        .groupby('network_src').mean('region_src')
        .groupby('network_dst').mean('region_dst')
    )

    return dataset
