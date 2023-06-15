from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import TimeseriesAggregator, ConnectivityExtractor, Parcellation


@dataclass
class ConnectivityPipeline(TransformerMixin, BaseEstimator):
    """Parcellate regions, aggregate networks, and extract connectivity."""

    atlas: str = 'dosenbach2010'
    kind: str = 'correlation'

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018/derivatives/resting_timeseries/'
    region_to_network: pd.DataFrame = None  # map regions to groups (e.g., the same regions, networks, random networks)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):

        parcellation = Parcellation(
            self.atlas,
            bids_dir=self.bids_dir,
            cache_dir=self.parcellation_cache_dir)

        # if no network mapping is provided, use the one from the parcellation
        if (self.region_to_network is None) or (len(self.region_to_network) == 0):
            self.region_to_network = parcellation.labels_['network'].to_dict()

        pipe = Pipeline([
            ('parcellation', parcellation),
            ('timeseries_aggregation', TimeseriesAggregator(mapping=self.region_to_network)),
            ('connectivity', ConnectivityExtractor(self.kind))
            # TODO add support for *_connectivity aggregations

        ])

        conn = pipe.fit_transform(X)
        nodes = pipe.named_steps['timeseries_aggregation'].groups_

        # TODO if we are aggregating connectivity, we need to do get node names from there

        self.dataset_ = xr.DataArray(
            conn,
            coords={'subject': parcellation.dataset_['subject'],
                    'node': nodes},
            dims=['subject', 'node', 'node'],
            name='connectivity')

        # select only queried subjects
        if X is not None:
            selected_subjects = X.reshape(-1).tolist()
            self.dataset_ = self.dataset_.sel(dict(subject=selected_subjects))

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
