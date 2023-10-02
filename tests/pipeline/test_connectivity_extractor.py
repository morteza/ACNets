import pytest
from src.acnets.pipeline import Parcellation, ConnectivityExtractor


# 'covariance', 'correlation', 'partial correlation',
# 'chatterjee', 'transfer_entropy', 'tangent', 'precision'

@pytest.mark.parametrize('atlas_name,kind', [
    ('dosenbach2007', 'tangent'),
    ('dosenbach2010', 'correlation'),
    ('difumo_64_2mm', 'chatterjee'),
    ('difumo_64_2mm', 'tangent'),
])
def test_connectivity_extractor(atlas_name, kind):

  dataset = Parcellation(atlas_name=atlas_name).fit().dataset
  timeseries = dataset['timeseries'].values

  n_subjects = dataset.coords['subject'].shape[0]
  n_regions = dataset.coords['region'].shape[0]

  extractor = ConnectivityExtractor(kind=kind, verbose=1)
  conn = extractor.fit_transform(timeseries)

  assert conn.shape == (n_subjects, n_regions, n_regions)
