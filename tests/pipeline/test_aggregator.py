import numpy as np
import pandas as pd
import pytest

from acnets.pipeline import Parcellation, TimeseriesAggregator


@pytest.mark.parametrize('atlas_name', [
    'dosenbach2010',
    # 'gordon2016_2mm',
    # 'difumo64_2mm'
])
def test_network_aggregator(atlas_name):

    # TimeseriesAggregator took a `mapping` DataFrame when this was written; it
    # now reads the `network` coordinate off the dataset and is selected with
    # `strategy`. Both it and Parcellation deal in xarray Datasets, so the shape
    # assertion has to go through the `timeseries` variable rather than the
    # Dataset, which has no .shape.
    parcellation = Parcellation(atlas_name=atlas_name)
    X = parcellation.fit_transform('all')

    aggregator = TimeseriesAggregator(strategy='network')
    X_agg = aggregator.fit_transform(X)

    n_subjects = X.sizes['subject']
    n_timepoints = X.sizes['timepoint']
    n_networks = parcellation.labels_['network'].nunique()

    assert X_agg['timeseries'].shape == (n_subjects, n_timepoints, n_networks)
    assert X_agg.attrs['timeseries_aggregation_strategy'] == 'network'
    # the region-level signal is kept alongside, not thrown away
    assert X_agg['region_timeseries'].shape == (n_subjects, n_timepoints,
                                                len(parcellation.labels_))


def test_connectivity_aggregator_algorithm():
    """standalone implementation of the connectivity matrix aggregation.

    This is the core algorithm that is implemented in the ConnectivityAggregator
    class to aggregate region-level connectivity matrices into networks, random
    networks, or the same regions.

    """

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

    # stack rather than reset_index().melt(): with a MultiIndex on the columns
    # too, reset_index names the inserted columns ('region_row', '') and the
    # melt cannot find its id_vars.
    X_long = (X_regions
              .stack(['region_col', 'group_col'], future_stack=True)
              .rename('connectivity')
              .reset_index())

    X_networks = X_long.groupby(['group_row', 'group_col'], sort=True)
    X_networks = X_networks['connectivity'].mean()

    X_networks = X_networks.reset_index().pivot(index='group_row',
                                                columns='group_col',
                                                values='connectivity')

    assert X_networks.shape == (n_networks, n_networks)
    # aggregation is a mean over the region pairs, so it stays in range
    assert X_networks.to_numpy().min() >= X.min()
    assert X_networks.to_numpy().max() <= X.max()
