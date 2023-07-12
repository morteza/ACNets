from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from typing import Callable, Literal


class ConnectivityAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level connectivity into networks, random networks, or the same regions."""

  def __init__(self,
               aggregation_strategy: Literal[None, 'network', 'random_network'] = None,
               reduce_fn: Callable = np.mean,
               ) -> None:

    self.aggregation_strategy = aggregation_strategy

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):
    self.dataset_ = dataset
    self.node_type = dataset['timeseries'].dims[-1]

    if self.aggregation_strategy is not None and self.node_type != 'region':
      raise ValueError(f'Time-series are already aggregated. '
                       f'Connectivity aggregation strategy `{self.aggregation_strategy}` is not supported.')

    return self

  def transform(self, dataset):
    return self.dataset_
