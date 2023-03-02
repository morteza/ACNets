import pandas as pd
import numpy as np
import xarray as xr
from dataclasses import dataclass
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline

from . import CerebellumParcellation, ConnectivityExtractor
from os import PathLike


@dataclass
class CerebellumPipeline(TransformerMixin, BaseEstimator):
    """Parcellate regions, extract cerebellum, and calculate connectivity."""

    difumo_dimension: int = 64
    kind: str = 'tangent'
    agg_networks: bool = True
    mock: bool = False

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018_resting'

    def __post_init__(self):
        if self.mock:
            self.transform = self.mock_transform

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        self.p = CerebellumParcellation(
            self.difumo_dimension,
            bids_dir=self.bids_dir,
            cache_dir=self.parcellation_cache_dir)

        self.c = ConnectivityExtractor(self.kind)

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
