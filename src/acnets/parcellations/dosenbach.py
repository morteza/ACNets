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

  if atlas_name == 'dosenbach2007':
    return load_dosenbach2007_masker(t_r=t_r)
  elif atlas_name == 'dosenbach2010':
    return load_dosenbach2010_masker(t_r=t_r)


def load_dosenbach2007_masker(atlas_labels_path='data/dosenbach2007/ROIs.csv', t_r=None):

  # TODO put atlas_labels in this file instead of relying on  external file

  atlas_labels = pd.read_csv(atlas_labels_path)
  atlas_labels.set_index('region', inplace=True)

  coords = atlas_labels[['x', 'y', 'z']].values

  masker = maskers.NiftiSpheresMasker(
      seeds=coords,
      smoothing_fwhm=6,
      radius=5,
      allow_overlap=False,
      detrend=True,
      standardize=True,
      low_pass=0.08,
      high_pass=0.009,
      t_r=t_r,
      verbose=0)

  return masker, atlas_labels


def load_dosenbach2010_masker(t_r=None):
  atlas_dict = datasets.fetch_coords_dosenbach_2010(
      ordered_regions=False,
      legacy_format=False)

  atlas_dict.pop('description', None)
  atlas_coords = atlas_dict['rois'].values
  atlas_dict['networks'] = atlas_dict['networks'].reset_index()

  masker = maskers.NiftiSpheresMasker(
      seeds=atlas_coords,
      smoothing_fwhm=6,
      radius=5,
      allow_overlap=False,
      #   detrend=True,
      #   standardize=True,
      #   low_pass=0.08,
      #   high_pass=0.009,
      t_r=t_r,
      verbose=0)

  atlas_labels = pd.concat([pd.DataFrame(v) for _, v in atlas_dict.items()],
                           ignore_index=False,
                           axis=1)

  atlas_labels.rename(columns={0: 'region'}, inplace=True)
  atlas_labels.set_index('region', inplace=True)

  atlas_labels = atlas_labels.reindex(atlas_dict['labels'])

  # TODO return Bunch instead
  return masker, atlas_labels
