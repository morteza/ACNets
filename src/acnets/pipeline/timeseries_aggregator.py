from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable


class TimeseriesAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal['region', 'network', 'random_network'] = 'region',
               reduce_fn: Callable = np.mean,
               ) -> None:

    self.strategy = strategy

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):

    self.dataset_ = dataset

    if self.strategy == 'region':
      return self

    if self.strategy == 'random_network':
      self.dataset_['network'] = (['region'], np.random.permutation(self.dataset_['network']))

    # either 'network' or 'random_network'
    network_timeseries = self.dataset_.groupby('network').mean(dim='region')['timeseries']
    network_timeseries = network_timeseries.transpose('subject', 'timepoint', 'network')
    self.dataset_['timeseries'] = network_timeseries

    return self

  def transform(self, dataset):

    return self.dataset_
