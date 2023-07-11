import pytest
from src.acnets.pipeline import Parcellation
from src.acnets.pipeline import TimeseriesAggregator
import pandas as pd
import numpy as np


@pytest.mark.parametrize('atlas_name', [
    'dosenbach2010',
    # 'gordon2016_2mm',
    # 'difumo64_2mm'
])
def test_network_aggregator(atlas_name):

    parcellation = Parcellation(atlas_name=atlas_name)
    X = parcellation.fit_transform(None)

    region_to_network = parcellation.labels_['network'].to_dict()
    region_to_network = pd.DataFrame.from_dict(region_to_network, orient='index', columns=['group'])

    aggregator = TimeseriesAggregator(mapping=region_to_network)
    X_agg = aggregator.fit_transform(X)

    n_subjects = X.shape[0]
    n_timepoints = X.shape[1]
    n_networks = parcellation.labels_['network'].nunique()
    assert X_agg.shape == (n_subjects, n_timepoints, n_networks)


def test_connectivity_aggregator_algorithm():

    n_regions = 100
    n_networks = 5

    X = np.random.rand(n_regions, n_regions)
    groups = np.random.randint(0, n_networks, n_regions)

    mapping = pd.DataFrame({'region': np.arange(0, n_regions), 'group': groups})
    mapping['group'] = mapping['group'].apply(lambda x: f'grp_{x}')
    mapping.set_index(['region', 'group'], inplace=True)

    X_regions = pd.DataFrame(X, index=mapping.index.copy(), columns=mapping.index.copy())
    X_regions.index.names = ['region_row', 'group_row']
    X_regions.columns.names = ['region_col', 'group_col']

    X_regions = X_regions.reset_index().melt(id_vars=['region_row', 'group_row'],
                                             value_name='connectivity')

    X_networks = X_regions.groupby(['group_row', 'group_col'], sort=True)
    X_networks = X_networks['connectivity'].apply(lambda v: np.mean(v))

    X_networks = X_networks.reset_index().pivot(index='group_row', columns='group_col')

    assert X_networks.shape == (n_networks, n_networks)
