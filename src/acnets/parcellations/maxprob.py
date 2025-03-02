import pandas as pd
from nilearn import datasets, maskers


def load_masker(atlas_name, mask_img):
  atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
  atlas_labels_img = atlas.maps

  masker = maskers.NiftiLabelsMasker(
      labels_img=atlas_labels_img,
      mask_img=mask_img,
      # standardize=True,
      #  memory='tmp/nilearn_cache',
      verbose=0)

  atlas_labels = pd.DataFrame(atlas.labels, columns=['region'])
  atlas_labels = atlas_labels.drop(index=[0]).set_index('region')

  # TODO return Bunch instead
  return masker, atlas_labels
