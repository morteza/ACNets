from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import TimeseriesAggregator, ConnectivityExtractor, Parcellation, ConnectivityAggregator


@dataclass
class MultiScaleConnectivityPipeline(TransformerMixin, BaseEstimator):
    """Aggregation of several steps to extract region-level and network-level features.

    Steps include parcellation, aggregate timeseries, extract connectivity matrices,
    and aggregate connectivity matrices.

    Features are extracted at two scales: region-level and network-level. Region-level features
    incudes mean and standard deviation of timeseries. Network-level features include within-network
    and between-network connectivity.

    Parameters
    ----------
    atlas : str, default='dosenbach2010'
        Name of the atlas to use for parcellation.
    kind : str, default='correlation'
        Type of connectivity to extract. See `acnets.pipeline.ConnectivityExtractor`.
    bids_dir : PathLike, default='data/julia2018'
        Path to BIDS directory.
    parcellation_cache_dir : PathLike, default='data/julia2018/derivatives/resting_timeseries/'
        Path to directory to cache parcellation results.

    Attributes
    ----------
    dataset_ : xarray.DataArray
        Dataset with timeseries and connectivity features.

    Returns
    -------
    features : np.ndarray
        Concatenation of region-level and network-level features.

    """

    atlas: str = 'dosenbach2010'
    kind: str = 'partial correlation'

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018/derivatives/resting_timeseries/'

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):

        datasets = {}
        features = {}
        # parcellation
        parcellation = Parcellation(self.atlas, bids_dir=self.bids_dir,
                                    cache_dir=self.parcellation_cache_dir)

        datasets['parcellation'] = parcellation.fit_transform(None)
        features['regions'] = datasets['parcellation']['timeseries'].mean('timepoint')

        # region-level time-series
        ts_agg = TimeseriesAggregator(strategy=None)
        ds_regions = ts_agg.fit_transform(ds_parcellation)

        # connectivity extraction
        conn_extractor = ConnectivityExtractor(kind=self.kind)
        dataset = conn_extractor.fit_transform(ds_regions)

        # connectivity aggregation
        conn_agg = ConnectivityAggregator(strategy='network')
        dataset = conn_agg.fit_transform(ds_regions)


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
