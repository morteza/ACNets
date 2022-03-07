from nilearn import datasets, maskers


def load_masker(atlas_name, mask_img, t_r=3.0):
  """
  Load a masker for the given atlas.

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

  atlas_labels = datasets.fetch_coords_dosenbach_2010(
      ordered_regions=True,
      legacy_format=False)
  atlas_coords = atlas_labels['rois'].values

  masker = maskers.NiftiSpheresMasker(
      seeds=atlas_coords,
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