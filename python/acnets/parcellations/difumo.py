import pandas as pd
from nilearn import datasets, plotting, maskers


def load_masker(atlas_name, mask_img):
  _, dim, res = atlas_name.split('_')
  atlas = datasets.fetch_atlas_difumo(
      dimension=dim,
      resolution_mm=res,
      legacy_format=False)

  masker = maskers.NiftiMapsMasker(
      atlas.maps,
      mask_img=mask_img,
      detrend=True,
      standardize=True,
      verbose=0)

  atlas_coordinates = (
      plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas.maps))
  atlas_labels = pd.concat([atlas.labels, pd.DataFrame(atlas_coordinates)], axis=1)
  atlas_labels.rename(columns={
      0: 'mni152_x',
      1: 'mni152_y',
      2: 'mni152_z'}, inplace=True)

  atlas_labels.drop(columns=['component'], inplace=True)
  atlas_labels.index.name = 'region'

  return masker, atlas_labels
