from dataclasses import dataclass
from os import PathLike
from typing import Iterable

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline, make_union, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from . import (ConnectivityAggregator, ConnectivityExtractor, Parcellation,
               TimeseriesAggregator)


class ExtractH1Features(TransformerMixin, BaseEstimator):
    """Extract region-level timeseries features (H1).

    Currently, only mean of timeseries is extracted.

    Inputs
    ------
    dataset['timeseries']
        shape: (timepoint, region)

    """

    def __init__(self):
        pass

    def fit(self, dataset, y=None):

        self.feature_names = dataset['timeseries'].coords['region'].values

        return self

    def transform(self, dataset):

        features = dataset['timeseries'].mean('timepoint').values
        return features

    def get_feature_names_out(self, input_features):
        return self.feature_names

    @classmethod
    def get_pipeline(cls):
        pipe = Pipeline([
            ('h1_features', ExtractH1Features()),
            # TODO normalize timeseries
        ])
        return pipe


class ExtractH2Features(TransformerMixin, BaseEstimator):
    """Extract within-network connectivity features (H2).

    Inputs
    ------
    dataset['connectivity']
    TimeseriesAggregator
    ConnectivityExtractor
    ConnectivityAggregator

    """

    def __init__(self):
        pass

    def fit(self, dataset, y=None):
        return self

    def transform(self, dataset):
        node_type = dataset['connectivity'].dims[-1]
        self.feature_names = dataset['connectivity'].coords[node_type].values.tolist()

        conn_vec = np.array([np.diag(conn)
                             for conn in dataset['connectivity'].values])

        return conn_vec

    def get_feature_names_out(self, input_features):
        return self.feature_names

    @classmethod
    def get_pipeline(cls, kind='partial correlation'):
        pipe = Pipeline([
            ('aggregate_timeseries', TimeseriesAggregator(strategy=None)),
            ('extract_connectivity', ConnectivityExtractor(kind=kind)),
            ('aggregate_connectivity', ConnectivityAggregator(strategy='network')),
            ('h2_features', ExtractH2Features())
        ])
        return pipe


class ExtractH3Features(TransformerMixin, BaseEstimator):
    """Extract between-network connectivity features (H3).

    Inputs
    ------
    dataset['connectivity']
    TimeseriesAggregator
    ConnectivityExtractor

    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, dataset, y=None):
        node_type = dataset['connectivity'].dims[-1][:-4]

        n_features = dataset['connectivity'].values.shape[1]
        # extract feature names
        self.feature_names = pd.DataFrame(
            data=np.zeros((n_features, n_features)),
            columns=dataset[node_type + '_src'],
            index=dataset[node_type + '_dst'])

        sep = ' \N{left right arrow} '
        self.feature_names = (self.feature_names
                                  .stack().to_frame()
                                  .apply(
                                      lambda x: sep.join(x.name), axis=1)
                                  .unstack()).values
        self.feature_names = self.feature_names[np.triu_indices(self.feature_names.shape[0],
                                                                k=self.k)].tolist()

        return self

    def transform(self, dataset):
        conns = dataset['connectivity'].values
        conn_vectorized = np.array([conn[np.triu_indices(conn.shape[0], k=self.k)]
                                    for conn in conns])

        return conn_vectorized

    def get_feature_names_out(self, input_features):
        return self.feature_names

    @classmethod
    def get_pipeline(cls, kind='partial correlation', k=1):
        # H3: between-network connectivity
        # non-diagonal connectivity between networks (shape: N_networks * N_networks / 2)
        pipe = Pipeline([
            ('aggregate_timeseries', TimeseriesAggregator(strategy='network')),
            ('extract_connectivity', ConnectivityExtractor(kind=kind)),
            ('h3_features', ExtractH3Features(k))
        ])
        return pipe


@dataclass
class MultiScaleClassifier(Pipeline):
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
    kind : str, default='partial correlation'
        Type of connectivity to extract. See `acnets.pipeline.ConnectivityExtractor` for choices.
    bids_dir : PathLike, defaults to 'data/julia2018'.
        Path to BIDS directory. if you are using Ray Tune, set this to absolute path.
    parcellation_cache_dir : PathLike, defaults to 'data/julia2018/derivatives/resting_timeseries/'.
        Path to directory to cache parcellation results. if you are using Ray Tune, set this to absolute path.

    Attributes
    ----------
    dataset_ : xarray.DataArray
        Dataset with timeseries and connectivity features.

    Returns
    -------
    y_pred : array-like
        Predicted labels.

    """

    def __init__(self,
                 atlas: str = 'dosenbach2010',
                 kind: str = 'partial correlation',
                 k=1,
                 classifier=XGBClassifier(),
                 memory=None, verbose=False):

        self.atlas = atlas
        self.kind = kind
        self.k = k
        self.classifier = classifier
        self.memory = memory
        self.verbose = verbose

        super().__init__(self.get_classification_steps(), memory=memory, verbose=verbose)

    def get_classification_steps(self):

        feature_extractors = [
            ('h1', ExtractH1Features.get_pipeline()),
            ('h2', ExtractH2Features.get_pipeline(kind=self.kind)),
            ('h3', ExtractH3Features.get_pipeline(kind=self.kind, k=self.k))
        ]

        steps = [
            ('parcellation', Parcellation(atlas_name=self.atlas)),
            ('extract_features', FeatureUnion(feature_extractors)),
            ('scale', StandardScaler()),
            ('zerovar', VarianceThreshold()),
            ('clf', self.classifier)
        ]

        return steps

    def get_feature_extractor(self):
        return self[:2]

    def get_classification_head(self):
        return self[2:]

    def get_feature_names_out(self, input_features=None):
        return self.get_feature_extractor().get_feature_names_out(input_features)

    def __getitem__(self, ind):
        """This is a slight modification of `Pipeline.__getitem__` to avoid copying the steps.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return Pipeline(
                self.steps[ind], memory=self.memory, verbose=self.verbose
            )
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    def set_params(self, **kwargs):

        return super().set_params(**kwargs)
