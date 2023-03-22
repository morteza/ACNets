from nilearn import datasets, maskers
import pandas as pd


def load_masker(atlas_name: str, mask_img=None, t_r=3.0):
  """
  Load a masker for the given atlas.

  TODO: t_r must be set for the butterworth filtering.

  Parameters
  ----------
  atlas_name : str
      Atlas name.
  mask_img : Niimg-like object
      See http://nilearn.github.io/manipulating_images/input_output.html
      The mask.

  Returns
  -------
  masker : instance of NiftiLabelsMasker
      The masker.
  atlas_labels : pandas.DataFrame
      List of regions in the atlas.
  """

  return load_fiedman2020_masker(t_r=t_r)


def load_fiedman2020_masker(atlas_labels_path='data/friedman2020/ROIs.csv', t_r=None):

  # TODO put atlas_labels in this file instead of relying on  external file

  atlas_labels = pd.read_csv(atlas_labels_path)
  atlas_labels.set_index('region', inplace=True)

  coords = atlas_labels[['x', 'y', 'z']].values

  masker = maskers.NiftiSpheresMasker(
      seeds=coords,
      smoothing_fwhm=6,
      radius=8,
      allow_overlap=True,
      #   detrend=False,
      #   standardize=False,
      #   low_pass=0.08,
      #   high_pass=0.009,
      t_r=t_r,
      verbose=0)

  return masker, atlas_labels
