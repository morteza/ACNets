import logging


def get_tr(self, default=3.0):
  """ use pybids to extract TR from BIDS metadata.

  Note:  if error occures while loadding the BIDS dataset, the default value will be returned.

  Returns
  -------
  int
      Repetition time in seconds
  """

  try:
    from bids import BIDSLayout
    layout = BIDSLayout(self.fmriprep_dir, derivatives=True)
    t_r = layout.get_tr(derivatives=True, task='rest')
    return t_r
  except Exception as e:
    logging.warn('Error occurred while retrieving TR from the BIDS dataset.', e)
    return default

