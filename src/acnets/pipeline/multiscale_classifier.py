from dataclasses import dataclass
from os import PathLike

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from . import (ConnectivityAggregator, ConnectivityExtractor, Parcellation,
               TimeseriesAggregator)


class ExtractH1Features(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, dataset, y=None):
        return self

    def transform(self, dataset):
        self.feature_names = dataset['timeseries'].coords['region'].values

        features = dataset['timeseries'].mean('timepoint').values

        return features

    def get_feature_names_out(self, input_features):
        return self.feature_names

    @classmethod
    def get_pipeline(cls):
        pipe = Pipeline([
            ('extract_features', ExtractH1Features()),
            # TODO normalize timeseries
        ])
        return pipe


class ExtractH2Features(TransformerMixin, BaseEstimator):
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
    def get_pipeline(cls):
        # H2
        # within-network connectivity
        pipe = Pipeline([
            ('aggregate_ts', TimeseriesAggregator(strategy=None)),
            ('extract_conn', ConnectivityExtractor(kind='partial correlation')),
            ('aggregate_conn', ConnectivityAggregator(strategy='network')),
            ('extract_features', ExtractH2Features())
        ])
        return pipe


class ExtractH3Features(TransformerMixin, BaseEstimator):
    def __init__(self, k=0):
        self.k = k

    def fit(self, dataset, y=None):
        return self

    def transform(self, dataset):
        conns = dataset['connectivity'].values
        conn_vectorized = np.array([conn[np.triu_indices(conn.shape[0], k=self.k)]
                                    for conn in conns])

        node_type = dataset['connectivity'].dims[-1][:-4]

        self.feature_names = pd.DataFrame(
            data=np.zeros((conns.shape[1], conns.shape[2])),
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

        return conn_vectorized

    def get_feature_names_out(self, input_features):
        return self.feature_names

    @classmethod
    def get_pipeline(cls):
        # H3: between-network connectivity
        # non-diagonal connectivity between networks (shape: N_networks * N_networks / 2)
        pipe = Pipeline([
            ('aggregate_ts', TimeseriesAggregator(strategy='network')),
            ('extract_conn', ConnectivityExtractor(kind='partial correlation')),
            ('extract_features', ExtractH3Features())
        ])
        return pipe


@dataclass
class MultiScaleClassifier(TransformerMixin, BaseEstimator):
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
    features : np.array
        Concatenation of region-level and network-level features.

    """

    atlas: str = 'dosenbach2010'
    kind: str = 'partial correlation'

    #  if you are using Ray Tune, set these params to absolute paths.
    bids_dir: PathLike = 'data/julia2018'
    parcellation_cache_dir: PathLike = 'data/julia2018/derivatives/resting_timeseries/'

    def fit(self, X, y=None, **fit_params):
        feature_extractors = {
            'h1': ExtractH1Features.get_pipeline(),
            'h2': ExtractH2Features.get_pipeline(),
            'h3': ExtractH3Features.get_pipeline()
        }

        self.model = Pipeline([
            ('parcellation', Parcellation(atlas_name='dosenbach2010')),
            ('extract_features', make_union(*feature_extractors.values())),
            ('scale', StandardScaler()),
            ('zerovar', VarianceThreshold()),
            ('clf', XGBClassifier())
            # ('ica', FastICA(n_components=20)),
            # ('select', SelectFromModel(RandomForestClassifier(),
            #                            max_features=lambda x: min(10, x.shape[1]))),
            # ('clf', RandomForestClassifier())
            # ('select', SelectFromModel(LinearSVC(penalty='l2', dual=False, max_iter=10000),
            #                            max_features=lambda x: min(10, x.shape[1]))),
            # ('clf', LinearSVC(penalty='l2', dual=False, max_iter=10000))
        ])
        self.model.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def get_feature_names_out(self, input_features):
        return self.model.get_feature_names_out(input_features)

    def score(self, X, y=None):
        return self.model.score(X, y)

    def get_encoder(self):
        return self.model[:2]

    def get_classification_head(self):
        return self.model[2:]
