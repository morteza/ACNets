from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Literal, Callable


class ConnectivityAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level connectivity into networks, random networks, or the same regions."""

  def __init__(self,
               mapping: pd.DataFrame = None,
               reduce_fn: Callable = np.mean,
               ) -> None:

    if (mapping is None) or (len(mapping) == 0):
      raise ValueError('Mappings must be provided.')

    self.mapping = mapping.copy()

    self.groups_ = self.mapping['group'].unique().tolist()

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, X=None, y=None, **fit_params):
    return self

  def transform(self, X):
    conn = []
    for X_subj in X:
        X_df = pd.DataFrame(X)
        X_df.index = [f'grp_{i}' for i in range(len(X_df.index))]
        X_df.columns = [f'grp_{i}' for i in range(len(X_df.columns))]

        self.mapping['connectivity'] = [x for x in X_subj]
        ts = self.mapping.groupby('group')['connectivity'].apply(lambda ts: self.reduce_fn(ts))
        ts_arr = np.asarray(ts.to_list())
        conn.append(ts_arr)


    return self.timeseries_
