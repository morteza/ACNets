import pytest


def test_load_julia2018_connectivity(connectivity_parcellations, connectivity_measures):
  from src.acnets.datasets import load_julia2018_connectivity

  for p in connectivity_parcellations:
    for m in connectivity_measures:
      load_julia2018_connectivity(parcellation=p, kind=m)
