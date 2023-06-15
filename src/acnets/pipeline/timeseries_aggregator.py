from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable


class TimeseriesAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               region_to_network: pd.DataFrame = None,
               reduce_fn: Callable = np.mean,
               ) -> None:

    if (region_to_network is None) or (len(region_to_network) == 0):
      raise ValueError('Mappings must be provided.')

    self.region_to_network = region_to_network.copy()

    self.groups_ = self.region_to_network['group'].unique().tolist()

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, X=None, y=None, **fit_params):
    return self

  def transform(self, X):
    timeseries = []
    for X_subj in X:
      self.region_to_network['timeseries'] = [x for x in X_subj.T]
      ts = self.region_to_network.groupby('group')['timeseries'].apply(lambda ts: self.reduce_fn(ts))
      ts_arr = np.asarray(ts.to_list()).T
      timeseries.append(ts_arr)

    self.timeseries_ = np.asarray(timeseries)  # shape: (n_subjects, n_timepoints, n_groups)

    return self.timeseries_
