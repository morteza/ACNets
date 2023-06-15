import pytest
from src.acnets.pipeline import Parcellation
from src.acnets.pipeline import TimeseriesAggregator
import pandas as pd


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
