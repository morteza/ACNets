import pytest
from python.acnets.pipeline.parcellation import Parcellation


@pytest.mark.parametrize('dimension', [64, 128])
@pytest.mark.parametrize('resolution_mm', [2])
def test_difumo_parcellation(dimension, resolution_mm):
  atlas_name = f'difumo_{dimension}_{resolution_mm}'

  dataset = Parcellation(atlas_name=atlas_name).fit().dataset
  assert dataset['timeseries'].shape == (34, dimension, 125)
