from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable


class TimeseriesAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal[None, 'network', 'random_network'] = None,  # None = 'region'
               ) -> None:

    self.strategy = strategy

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):

    return self

  def transform(self, dataset):

    self.dataset_ = dataset

    if self.strategy is None:
      return self.dataset_

    if self.strategy == 'random_network':
      self.dataset_['network'] = (['region'],
                                  np.random.permutation(self.dataset_['network']))

    # either 'network' or 'random_network'
    network_timeseries = self.dataset_.groupby('network').mean(dim='region')['timeseries']
    network_timeseries = network_timeseries.transpose('subject', 'timepoint', 'network')
    self.dataset_['region_timeseries'] = self.dataset_['timeseries']
    self.dataset_['timeseries'] = network_timeseries

    return self.dataset_

  def get_feature_names_out(self, input_features=None):
    if self.strategy is None:
      return input_features
    return None
