# to combine region-level connectivities into confirmatory network-level factors

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class NetworkAggregator(TransformerMixin, BaseEstimator):
  def __init__(self,
               atlas_labels,
               method=np.mean,
               ) -> None:

    if 'network' not in atlas_labels.columns:
      raise ValueError('atlas_labels must indexed by regions and have a column named "network"')

    self.atlas_labels = atlas_labels
    self.networks_ = list(self.atlas_labels.groupby('network').groups.keys())

    if method == 'mean':
      self.method = np.mean
    elif method == 'median':
      self.method = np.median
    elif callable(method):
      self.method = method
    else:
      raise ValueError(f'Method {method} not supported.')

    super().__init__()

  def fit(self, X=None, y=None, **fit_params):
    return self

  def transform(self, X):
    timeseries = []
    for X_subj in X:
      regions_df = self.atlas_labels.copy()
      regions_df['timeseries'] = [x for x in X_subj.T]
      ts = regions_df.groupby('network')['timeseries'].apply(lambda ts: self.method(ts))
      ts_arr = np.asarray(ts.to_list()).T
      timeseries.append(ts_arr)

    self.timeseries_ = np.asarray(timeseries)

    return self.timeseries_
