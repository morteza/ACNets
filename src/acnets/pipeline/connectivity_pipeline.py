import pandas as pd
import numpy as np
import xarray as xr
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from typing import Literal

from . import Parcellation, Aggregator, ConnectivityExtractor
from os import PathLike


@dataclass
class ConnectivityPipeline(TransformerMixin, BaseEstimator):
    """Parcellate regions, aggregate networks, and extract connectivity."""

    atlas: str = 'dosenbach2010'
    kind: str = 'correlation'
    agg_method: Literal['region',
                        'network',
                        'random_network',
                        'region_connectivity',
                        'network_connectivity',
                        'random_network_connectivity'] = 'network'

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018/derivatives/resting_timeseries/'

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        self.p = Parcellation(self.atlas,
                              bids_dir=self.bids_dir,
                              cache_dir=self.parcellation_cache_dir)
        self.n = Aggregator(self.p.labels_)
        self.c = ConnectivityExtractor(self.kind)

        if self.agg_method == 'region':
            conn = make_pipeline(self.p, self.c).fit_transform(X)
            nodes = self.p.labels_.index.to_list()
        elif self.agg_method == 'network':
            conn = make_pipeline(self.p, self.n, self.c).fit_transform(X)
            nodes = self.n.networks_
        elif self.agg_method == 'random_network':
            conn = make_pipeline(self.p, self.n, self.c).fit_transform(X)
            nodes = self.n.networks_
            raise NotImplementedError
        elif self.agg_method == 'region_connectivity':
            conn = make_pipeline(self.p, self.c).fit_transform(X)
            nodes = self.p.labels_.index.to_list()
            raise NotImplementedError
        elif self.agg_method == 'network_connectivity':
            conn = make_pipeline(self.p, self.n, self.c).fit_transform(X)
            nodes = self.n.networks_
            raise NotImplementedError
        elif self.agg_method == 'random_network_connectivity':
            conn = make_pipeline(self.p, self.n, self.c).fit_transform(X)
            nodes = self.n.networks_
            raise NotImplementedError

        self.dataset_ = xr.DataArray(
            conn,
            coords={'subject': self.p.dataset_['subject'],
                    'node': nodes},
            dims=['subject', 'node', 'node'],
            name='connectivity')

        # select only queried subjects
        if (X is not None) and (str(X) != 'all'):
            subjects_1d = X.reshape(-1).tolist()
            self.dataset_ = self.dataset_.sel(dict(subject=subjects_1d))

        return self.dataset_

    def get_feature_names_out(self,
                              input_features=None,
                              sep=' \N{left right arrow} '):

        if input_features is None:
            input_features = self.transform(None).coords['node'].values

        feature_names = pd.DataFrame(
            np.zeros((input_features.shape[0], input_features.shape[0])),
            columns=input_features, index=input_features)

        feature_names = feature_names.stack().to_frame().apply(
            lambda x:
            sep.join(x.name) if x.name[0] != x.name[1] else x.name[0],
            axis=1).unstack()
        return feature_names
