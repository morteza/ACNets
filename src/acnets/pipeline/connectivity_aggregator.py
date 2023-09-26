from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from typing import Callable, Literal


class ConnectivityAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level connectivity into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal[None, 'network', 'random_network'] = None,
               ) -> None:

    self.strategy = strategy

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):

    return self

  def transform(self, dataset):

    self.dataset_ = dataset

    if self.strategy is None:
      return self

    node_type = self.dataset_['timeseries'].dims[-1]

    # we need region-level connectivity matrices to aggregate
    if node_type != 'region':
      raise ValueError(f'Time-series are already aggregated. '
                       f'Connectivity aggregation strategy `{self.strategy}` is not supported.')

    if self.strategy == 'random_network':
      self.dataset_['network'] = (['region'],
                                  np.random.permutation(self.dataset_['network']))

    self.dataset_ = self.dataset_.assign_coords(network_src=('region_src', self.dataset_['network'].values))
    self.dataset_ = self.dataset_.assign_coords(network_dst=('region_dst', self.dataset_['network'].values))

    # defaults to take absolute value of connectivity matrices
    self.dataset_['connectivity'] = np.fabs(self.dataset_['connectivity'])

    self.dataset_['connectivity'] = (
        self.dataset_['connectivity']
        .groupby('network_src').mean('region_src')
        .groupby('network_dst').mean('region_dst')
    )
    return self.dataset_

  def get_feature_names_out(self, input_features=None):
    if self.strategy is None:
      return input_features
    elif 'network' in self.strategy:
      return self.dataset_.coords['network_src'].values.tolist()

    return None
