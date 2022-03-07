def load_masker(atlas_name, mask_img):
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
  atlas_labels : list of str
      List of labels in the atlas.
  """

  pass