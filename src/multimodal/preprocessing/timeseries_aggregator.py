from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import xarray as xr
from typing import Literal


class TimeseriesAggregator(TransformerMixin, BaseEstimator):
  """Aggregates region-level time-series into networks, random networks, or the same regions."""

  def __init__(self,
               strategy: Literal[None, 'network', 'random_network', 'wavelet'] = None,  # None = 'region'
               **kwargs
               ) -> None:

    self.strategy = strategy

    if strategy == 'wavelet':
      self._wavelet_name = kwargs.get('wavelet_name', 'db4')
      self._wavelet_coef_dim = kwargs.get('wavelet_coef_dim', -1)

    # the rest of init from scikit-learn
    super().__init__()

  def fit(self, dataset, y=None, **fit_params):

    return self

  def transform(self, dataset) -> xr.Dataset:

    new_dataset = dataset.copy()

    if self.strategy is None:
      return new_dataset

    if self.strategy == 'wavelet':
      import pywt
      ts = dataset['timeseries'].transpose('subject', 'region', 'timepoint')
      coefs = pywt.wavedec(ts, wavelet=self._wavelet_name)
      coefs_image = np.concatenate(coefs, axis=2)
      coefs_image = coefs_image[..., :self._wavelet_coef_dim]
      new_dataset['wavelets'] = (['subject', 'wavelet_dim', 'region'],
                                 coefs_image.transpose(0, 2, 1))  # subject, wavelet_dim, region
      return new_dataset

    # if strategy is 'network' or 'random_network'

    new_dataset = new_dataset.set_coords('network')

    if self.strategy == 'random_network':
      new_dataset['network'] = (['region'],
                                np.random.permutation(new_dataset['network']))

    # either 'network' or 'random_network'
    network_timeseries = new_dataset.groupby('network').mean(dim='region')['timeseries']
    network_timeseries = network_timeseries.transpose('subject', 'timepoint', 'network')
    new_dataset['region_timeseries'] = new_dataset['timeseries']
    new_dataset['timeseries'] = network_timeseries

    return new_dataset

  def fit_transform(self, X: xr.Dataset, y=None, **fit_params) -> xr.Dataset:
    if y is None:
        return self.fit(X, **fit_params).transform(X)
    else:
        return self.fit(X, y, **fit_params).transform(X)

  def get_feature_names_out(self, input_features=None):
    if self.strategy is None:
      return input_features
    return None
