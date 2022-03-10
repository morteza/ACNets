from nilearn import maskers, datasets, plotting

from python.acnets.pipeline.parcellation import Parcellation

from python.acnets.pipeline import Parcellation


def test_network_extraction():

  parcellation = Parcellation('dosenbach2010')
  parcellation.fit()

  return None
