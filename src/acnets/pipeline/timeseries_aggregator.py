from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable


class TimeseriesAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal[None, 'network', 'random_network'] = None,  # None = 'region'
               reduce_fn: Callable = np.mean,
               ) -> None:

    self.strategy = strategy

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, dataset):

    if self.strategy is None:
      return dataset

    if self.strategy == 'random_network':
      dataset['network'] = (['region'], np.random.permutation(dataset['network']))

    # either 'network' or 'random_network'
    network_timeseries = dataset.groupby('network').mean(dim='region')['timeseries']
    network_timeseries = network_timeseries.transpose('subject', 'timepoint', 'network')
    dataset['timeseries'] = network_timeseries

    return dataset
