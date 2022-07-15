import re
import pandas as pd
from nilearn import datasets, plotting, maskers


def load_masker(atlas_name: str, mask_img, t_r=3.0):
  dim, res = re.findall('difumo_(\\d{2,4})_(\\d)mm', atlas_name.lower())[0]

  atlas = datasets.fetch_atlas_difumo(
      dimension=int(dim),
      resolution_mm=int(res),
      legacy_format=False)

  masker = maskers.NiftiMapsMasker(
      atlas.maps,
      mask_img=mask_img,
      #   detrend=True,
      #   standardize=True,
      t_r=t_r,
      verbose=0)

  atlas_labels = atlas.labels
#   atlas_coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas.maps)
#   atlas_labels = pd.concat([atlas.labels, pd.DataFrame(atlas_coordinates)], axis=1)
  atlas_labels.rename(columns={
      0: 'x',
      1: 'y',
      2: 'z'}, inplace=True)

  atlas_labels.drop(columns=['component'], inplace=True)
  atlas_labels['network'] = atlas_labels['yeo_networks17']
  atlas_labels.rename(columns={'difumo_names': 'region'}, inplace=True)
  atlas_labels.set_index('region', inplace=True)

  return masker, atlas_labels
