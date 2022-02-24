import pytest
from python.acnets.datasets import load_connectivity


def test_load_connectivity(connectivity_parcellations, connectivity_measures):
  for p in connectivity_parcellations:
    for m in connectivity_measures:  
      load_connectivity(parcellation=p, kind=m)
