import re
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from nilearn.interfaces.bids import get_bids_files
from nilearn.interfaces.fmriprep import load_confounds_strategy
from tqdm import tqdm


from ..parcellations import maxprob, dosenbach, difumo, gordon, seitzman

_masker_funcs = {
    'cort-maxprob': maxprob.load_masker,
    'difumo': difumo.load_masker,
    'dosenbach': dosenbach.load_masker,
    'gordon': gordon.load_masker,
    'seitzman': seitzman.load_masker,
}


class Parcellation(TransformerMixin, BaseEstimator):
  """ to parcellate activities given an atlas.
  """

  def __init__(self,
               atlas_name='cort-maxprob-thr25-2mm',
               bids_dir='data/julia2018',
               denoise_strategy='simple',
               fmriprep_bids_space='MNI152NLin2009cAsym',
               cache_folder='data/julia2018_resting',
               factorize_networks=False,
               verbose=0) -> None:

    self.bids_dir = bids_dir
    self.verbose = verbose
    self.atlas_name = atlas_name
    self.cache_folder = cache_folder
    self.denoise_strategy = denoise_strategy
    self.factorize_networks = factorize_networks

    self.dataset_: xr.Dataset = None

    self.fmriprep_dir_ = Path(self.bids_dir) / 'derivatives/fmriprep_2020'
    self.fmriprep_bids_space = fmriprep_bids_space

    # validation (TODO intergate all the validations in a single function)
    if not self.cache_folder and not self.fmriprep_dir_.exists():
      raise ValueError('Neither BIDS dataset exists nor cached dataset is set.')

    for key, func in _masker_funcs.items():
      if re.match(key, atlas_name):
        self._load_masker = func

    self.masker_, self.labels_ = self._load_masker(self.atlas_name, None)

    super().__init__()

  def _get_fmriprep_files(self):
    # RESTING SCANS
    img_files = get_bids_files(
        self.fmriprep_dir_,
        file_tag='bold',
        modality_folder='func',
        filters=[('ses', 'rest'),
                 ('space', self.fmriprep_bids_space),
                 ('desc', 'preproc')],
        file_type='nii.gz')

    # BRAIN MASKS
    mask_files = get_bids_files(
        self.fmriprep_dir_,
        file_tag='mask',
        modality_folder='func',
        filters=[('ses', 'rest'),
                 ('space', self.fmriprep_bids_space),
                 ('desc', 'brain')],
        file_type='nii.gz')

    return img_files, mask_files

  def _load_masker(self, atlas_name, mask_file):
    # default implementation when none of the atlases matched in the __init__.
    raise NotImplementedError('Atlas name {} not recognized.'.format(self.atlas_name))

  def extract_timeseries(self, img_files, mask_files, atlas_name):

    _timeseries = {}

    confounds, sample_masks = load_confounds_strategy(img_files, self.denoise_strategy)

    # identify discarded scans
    n_scans = confounds[0].shape[0]
    valid_timepoints = set(np.hstack([s for s in sample_masks if s is not None]))
    timepoints_mask = np.zeros(n_scans, dtype=bool)
    timepoints_mask[np.array(list(valid_timepoints)) - 1] = True

    img_iterator = zip(img_files, mask_files, confounds, sample_masks)

    if self.verbose > 0:
      img_iterator = tqdm(img_iterator, total=len(img_files))

    # loop over all the subjects
    for img, mask, confound, sample_mask in img_iterator:

      subject = re.search('func/sub-(.*)_ses', img)[1]
      self.masker_, _ = self._load_masker(atlas_name, mask)
      ts = self.masker_.fit_transform(img, confound, sample_mask)

      if ts.shape[0] > len(valid_timepoints):
        ts = ts[timepoints_mask]

      _timeseries[subject] = ts

    return _timeseries

  def create_dataset(self, timeseries=None):

    # dims: (n_subjects, n_timepoints, n_regions)
    _timeseries_arr = np.stack(list(timeseries.values()))
    preprocessed_subjects = list(timeseries.keys())

    # atlas dataset
    atlas_ds = self.labels_.to_xarray()

    # time-series dataset
    timeseries_ds = xr.Dataset({
        'timeseries': (['subject', 'timepoint', 'region'], _timeseries_arr)
    }, coords={
        'timepoint': np.arange(1, _timeseries_arr.shape[1] + 1),
        'region': self.labels_.index,
        'subject': preprocessed_subjects,
    })

    # participants dataset
    # TODO path to the bids must be used here
    bids_participants = pd.read_csv('data/julia2018/participants.tsv', sep='\t')
    bids_participants.rename(columns={'participant_id': 'subject'}, inplace=True)
    # remove "sub-" prefix
    bids_participants['subject'] = bids_participants['subject'].apply(lambda x: x.split('-')[1])
    bids_participants.set_index('subject', inplace=True)
    bids_participants = bids_participants.query('index in @preprocessed_subjects')
    participants_ds = bids_participants.to_xarray()

    # merge arrays into a single dataset
    _ds = xr.merge([timeseries_ds, participants_ds, atlas_ds])

    return _ds

  def cache_dataset(self, overwrite=False):
    if not self.dataset_:
      raise ValueError('Parcellation has not been fitted yet.')

    if not self.cache_folder:
      raise ValueError('No cache_folder is set.')

    cached_ds_path = Path(self.cache_folder) / f'timeseries_{self.atlas_name}.nc'

    if overwrite or not cached_ds_path.exists():
      self.dataset_.to_netcdf(cached_ds_path, engine='netcdf4')

  def fit(self, X=None, y=None, **fit_params):  # noqa: N803

    # load from cache
    if self.cache_folder:
      cached_ds_path = Path(self.cache_folder) / f'timeseries_{self.atlas_name}.nc'
      if cached_ds_path.exists():
        self.dataset_ = xr.open_dataset(cached_ds_path)
        return self

    # fit if not already fitted
    if not self.dataset_:
      img_files, mask_files = self._get_fmriprep_files()
      time_series = self.extract_timeseries(img_files, mask_files, self.atlas_name)
      self.dataset_ = self.create_dataset(time_series)
      self.cache_dataset(overwrite=False)

    return self

  def transform(self, X=None):  # noqa: N803
    """_summary_

    Parameters
    ----------
    X : list, optional
        list of participant ids to select (BIDS-compatible without `sub-` prefix), e.g.,
        ['AVGP-01', 'NVGP-03'] returns two subjects. Passing `None` will return all
        the subjects. Defaults to None.

    Returns
    -------
    numpy.ndarray
        time-series of shape (n_size, n_timepoints, n_regions).

    Raises
    ------
    ValueError
        call fit() before calling transform().
    """
    if not self.dataset_:
      raise ValueError('Parcellation has not been fitted yet.')

    # FIXME
    ds = self.dataset_
    # if X is not None:
    #   _subjects = X.reshape(-1).tolist()
    #   ds = ds.sel(dict(subject=_subjects))

    timeseries = ds['timeseries'].values
    return timeseries
