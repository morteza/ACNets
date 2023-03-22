import pandas as pd
import numpy as np
import xarray as xr
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline

from . import Parcellation, NetworkAggregator, ConnectivityExtractor
from os import PathLike


@dataclass
class ConnectivityPipeline(TransformerMixin, BaseEstimator):
    """Parcellate regions, aggregate networks, and extract connectivity."""

    atlas: str = 'dosenbach2010'
    kind: str = 'correlation'
    agg_networks: bool = True
    mock: bool = False

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018/derivatives/resting_timeseries/'

    def __post_init__(self):
        if self.mock:
            self.transform = self.mock_transform

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        self.p = Parcellation(self.atlas,
                              bids_dir=self.bids_dir,
                              cache_dir=self.parcellation_cache_dir)
        self.n = NetworkAggregator(self.p.labels_)
        self.c = ConnectivityExtractor(self.kind)

        if self.agg_networks:
            conn = make_pipeline(self.p, self.n, self.c).fit_transform(X)
            nodes = self.n.networks_
        else:
            conn = make_pipeline(self.p, self.c).fit_transform(X)
            nodes = self.p.labels_.index.to_list()

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

    def mock_transform(self, X):

        n_features_dict = {
            ('gordon2014_2mm', True): 13,
            ('gordon2014_2mm', False): 333,
            ('dosenbach2010', True): 6,
            ('dosenbach2010', False): 160,
            ('difumo_64_2mm', True): 7,
            ('difumo_64_2mm', False): 64,
            ('seitzman2018', True): 14,
            ('seitzman2018', False): 300,
        }

        subjects = X
        if subjects is None:
            subjects = Parcellation(self.atlas).fit(None).dataset_['subject'].values

        n_features = n_features_dict[(self.atlas, self.agg_networks)]

        nodes = [f'node_{n}' for n in range(n_features)]

        mock_conn = np.random.rand(len(subjects), n_features, n_features)

        self.dataset_ = xr.DataArray(
            mock_conn,
            coords={'subject': subjects,
                    'node': nodes},
            dims=['subject', 'node', 'node'],
            name='connectivity')

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
