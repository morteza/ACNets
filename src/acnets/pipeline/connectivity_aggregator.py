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

    new_dataset = dataset

    if self.strategy is None:
      return self

    node_type = new_dataset['timeseries'].dims[-1]

    # we need region-level connectivity matrices to aggregate
    if node_type != 'region':
      raise ValueError(f'Time-series are already aggregated. '
                       f'Connectivity aggregation strategy `{self.strategy}` is not supported.')

    if self.strategy == 'random_network':
      new_dataset['network'] = (['region'],
                                  np.random.permutation(new_dataset['network']))

    new_dataset = new_dataset.assign_coords(network_src=('region_src', new_dataset['network'].values))
    new_dataset = new_dataset.assign_coords(network_dst=('region_dst', new_dataset['network'].values))

    # defaults to take absolute value of connectivity matrices
    new_dataset['connectivity'] = np.fabs(new_dataset['connectivity'])

    new_dataset['connectivity'] = (
        new_dataset['connectivity']
        .groupby('network_src').mean('region_src')
        .groupby('network_dst').mean('region_dst')
    )
    return new_dataset

  def get_feature_names_out(self, input_features=None):
    if self.strategy is None:
      return input_features
    elif 'network' in self.strategy:
      return self.dataset_.coords['network_src'].values.tolist()

    return None
