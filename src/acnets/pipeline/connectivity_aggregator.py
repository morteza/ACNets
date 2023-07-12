from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from typing import Callable


class ConnectivityAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level connectivity into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal['regions', 'networks', 'random'] = 'networks',
               reduce_fn: Callable = np.mean,
               ) -> None:

    if callable(reduce_fn):
      self.reduce_fn = reduce_fn
    else:
      raise ValueError(f'Reduction method {reduce_fn} not supported.')

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):
    return self

  def transform(self, dataset):
    return self.dataset_
