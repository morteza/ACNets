import pytest
from python.acnets.pipeline import Parcellation
from python.acnets.pipeline import NetworkAggregator


@pytest.mark.parametrize('atlas_name', [
    'gordon2014_2mm'
])
def test_network_aggregator(atlas_name):

    model = Parcellation(atlas_name=atlas_name, verbose=1).fit()
    X = model.transform(X=None)

    n_subjects = X.shape[0]
    n_timepoints = X.shape[1]
    n_networks = model.labels_['network'].nunique()

    network_model = NetworkAggregator(atlas_labels=model.labels_, method='mean')
    X_net = network_model.fit_transform(X)

    assert X_net.shape == (n_subjects, n_timepoints, n_networks)
