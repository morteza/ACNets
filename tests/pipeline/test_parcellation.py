import pytest
from src.acnets.pipeline.parcellation import Parcellation


@pytest.mark.parametrize('atlas_name,expected_regions_dim', [
    ('cort-maxprob-thr0-1mm', 48),
    ('cort-maxprob-thr25-2mm', 48),
    ('cort-maxprob-thr25-1mm', 48),
    ('dosenbach2007', 39),
    ('seitzman2018', 300),
    ('difumo_64_2mm', 64),
    ('difumo_128_2mm', 128),
    ('gordon2014_2mm', 333),
    ('dosenbach2010', 160)])
def test_parcellation(atlas_name, expected_regions_dim):

  n_timepoints = 124
  n_subjects = 32

  if 'difumo' in atlas_name:
    n_regions_in_label = int(atlas_name.split('_')[1])
    assert n_regions_in_label == expected_regions_dim

  model = Parcellation(atlas_name=atlas_name, verbose=1).fit()
  model.transform('all')
  dataset = model.dataset_
  # DEBUG model.labels_.to_csv('dosebach2010.csv')
  assert dataset['timeseries'].shape == (n_subjects, n_timepoints, expected_regions_dim)


def test_dosenbach2010_labels():

  from src.acnets.parcellations.dosenbach import load_dosenbach2010_masker

  _, labels = load_dosenbach2010_masker()
  assert 'network' in labels.columns
  assert 'networks' not in labels.columns
