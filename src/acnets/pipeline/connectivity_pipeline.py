from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import TimeseriesAggregator, ConnectivityExtractor, Parcellation, ConnectivityAggregator


@dataclass
class ConnectivityPipeline(TransformerMixin, BaseEstimator):
    """Aggregation of several steps to extract connectivity from timeseries.

    Steps include parcellation, aggregate timeseries, extract connectivity matrices,
    and aggregate connectivity matrices.

    Parameters
    ----------
    atlas : str, default='dosenbach2010'
        Name of the atlas to use for parcellation.
    kind : str, default='correlation'
        Type of connectivity to extract. See `acnets.pipeline.ConnectivityExtractor`.
    timeseries_aggregation : str, default=None
        Strategy to aggregate timeseries. See `acnets.pipeline.TimeseriesAggregator`.
    connectivity_aggregation : str, default=None
        Strategy to aggregate connectivity matrices. See `acnets.pipeline.ConnectivityAggregator`.
    bids_dir : PathLike, default='data/julia2018'
        Path to BIDS directory.

    Attributes
    ----------
    dataset_ : xarray.DataArray
        Dataset with timeseries and connectivity matrices.

    Returns
    -------
    conn : np.ndarray
        Connectivity matrices.

    """

    atlas: str = 'dosenbach2010'
    kind: str = 'correlation'
    timeseries_aggregation: Literal[None, 'network', 'random_network'] = None
    connectivity_aggregation: Literal[None, 'network', 'random_network'] = None

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):

        pipe = Pipeline([
            ('parcellation', Parcellation(self.atlas, bids_dir=self.bids_dir)),
            ('timeseries_aggregation', TimeseriesAggregator(strategy=self.timeseries_aggregation)),
            ('connectivity', ConnectivityExtractor(self.kind)),
            ('connectivity_aggregation', ConnectivityAggregator(strategy=self.connectivity_aggregation))
        ])

        self.dataset_ = pipe.fit_transform(X)

        # get connectivity matrices
        conn = self.dataset_['connectivity'].values

        return conn

    def get_feature_names_out(self,
                              input_features=None,
                              sep=' \N{left right arrow} '):

        # TODO: update to extract from dataset_

        if input_features is None:
            _ = self.transform(None)  # to initialize self.dataset_ if not already
            node_type = self.dataset_['connectivity'].dims[-1]
            input_features = self.dataset_.coords[node_type].values

        feature_names = pd.DataFrame(
            np.zeros((input_features.shape[0], input_features.shape[0])),
            columns=input_features, index=input_features)

        feature_names = feature_names.stack().to_frame().apply(
            lambda x:
            sep.join(x.name) if x.name[0] != x.name[1] else x.name[0],
            axis=1).unstack()
        return feature_names
