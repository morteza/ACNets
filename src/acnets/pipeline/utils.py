import logging
from joblib import Parallel
from sklearn import model_selection
from tqdm import tqdm


def get_tr(self, default=3.0):
  """ use pybids to extract TR from BIDS metadata.

  Note:  if error occurs while loadding the BIDS dataset, the default value will be returned.

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


class GridSearchCVProgressBar(model_selection.GridSearchCV):
    """Show progress bar in grid search (Borrowed from `pactools/grid_search.py`)."""

    def _get_param_iterator(self):

        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(Parallel):
            def __call__(self, iterable):
                bar = tqdm(total=max_value, title='GridSearchCV')
                iterable = bar(iterable)
                return super(ParallelProgressBar, self).__call__(iterable)

        model_selection._search.Parallel = ParallelProgressBar

        return iterator
