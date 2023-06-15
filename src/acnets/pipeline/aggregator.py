from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable, Dict


class Aggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               mappings: Dict[str, str] = None,
               method: Literal[
                   'region',
                   'network',
                   'random_network'] = 'network',
               reduce_fn: Callable = np.mean,
               ) -> None:

    if (mappings is None) or (len(mappings) == 0):
      raise ValueError('Mappings must be provided.')

    self.mappings = mappings

    if 'network' in method:
      self.mappings_df = pd.DataFrame.from_dict(mappings, orient='index', columns=['group'])
    elif 'region' in method:
      _m = {k: k for k in mappings.keys()}  # create identity region-to-region mapping
      self.mappings_df = pd.DataFrame.from_dict(_m, orient='index', columns=['group'])

    if 'random' in method:
      self.mappings_df['original_group'] = self.mappings_df['group']
      self.mappings_df['group'] = self.mappings_df['group'].sample(frac=1).values

    self.groups_ = self.mappings_df['group'].unique().tolist()

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, X=None, y=None, **fit_params):
    return self

  def transform(self, X):
    timeseries = []
    for X_subj in X:
      self.mappings_df['timeseries'] = [x for x in X_subj.T]
      ts = self.mappings_df.groupby('group')['timeseries'].apply(lambda ts: self.reduce_fn(ts))
      ts_arr = np.asarray(ts.to_list()).T
      timeseries.append(ts_arr)

    self.timeseries_ = np.asarray(timeseries)  # shape: (n_subjects, n_timepoints, n_groups)

    return self.timeseries_
