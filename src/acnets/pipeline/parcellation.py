import re
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from nilearn.interfaces.bids import get_bids_files
from nilearn.interfaces.fmriprep import load_confounds_strategy
from tqdm import tqdm


from ..parcellations import maxprob, dosenbach, difumo, gordon, seitzman, friedman, aal

_masker_funcs = {
    'cort-maxprob': maxprob.load_masker,
    'difumo': difumo.load_masker,
    'dosenbach': dosenbach.load_masker,
    'gordon': gordon.load_masker,
    'seitzman': seitzman.load_masker,
    'friedman': friedman.load_masker,
    'aal': aal.load_masker,
}


class Parcellation(TransformerMixin, BaseEstimator):
  """ parcellate voxel-wise timeseries for a given atlas.
  """

  def __init__(self,
               atlas_name='cort-maxprob-thr25-2mm',
               bids_dir=Path('data/julia2018'),
               denoise_strategy='simple',
               fmriprep_bids_space='MNI152NLin2009cAsym',
               normalize=False,
               verbose=0) -> None:

    self.atlas_name = atlas_name
    self.bids_dir = Path(bids_dir).expanduser()
    self.denoise_strategy = denoise_strategy
    self.cache_dir = self.bids_dir.expanduser() / 'derivatives/resting_timeseries/'
    self.normalize = normalize
    self.verbose = verbose

    self.fmriprep_dir_ = self.bids_dir / 'derivatives/fmriprep_2020'
    self.fmriprep_bids_space = fmriprep_bids_space

    # validation (TODO intergate all the validations in a single function)
    if not self.cache_dir and not self.fmriprep_dir_.exists():
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

    assert len(img_files) > 0, f'No resting scans found in {self.fmriprep_dir_}'

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

  def _normalize_func(self, x: xr.DataArray):
    """Normalize the subject data to [-1, 1] range."""
    x_std = (x - x.min(['subject'])) / (x.max(['subject']) - x.min(['subject']))
    x_std = x_std * 2 - 1
    return x_std

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

    if self.normalize:
      timeseries_ds['timeseries'] = self._normalize_func(timeseries_ds['timeseries'])

    # participants
    participants = pd.read_csv(Path(self.bids_dir).expanduser() / 'participants.tsv',
                               sep='\t')

    # remove "sub-" prefix
    participants['participant_id'] = \
        participants['participant_id'].apply(lambda x: x.replace('sub-', ''))
    participants = participants.query('index in @preprocessed_subjects')
    participants = participants.rename(columns={'participant_id': 'subject'})
    participants.set_index('subject', inplace=True)

    # merge data into a single dataset
    dataset = xr.merge([timeseries_ds, participants.to_xarray(), atlas_ds])

    return dataset

  def cache_dataset(self, dataset, overwrite=False):

    if not self.cache_dir:
      raise ValueError('`cache_dir` is not properly set.')

    cached_ds_path = Path(self.cache_dir).expanduser() / f'timeseries_{self.atlas_name}.nc5'

    if overwrite or not cached_ds_path.exists():
      dataset.to_netcdf(cached_ds_path, engine='h5netcdf')

  def fit(self, X=None, y=None, **fit_params):  # noqa: N803

    cached_ds_path = Path(self.cache_dir).expanduser() / f'timeseries_{self.atlas_name}.nc5'

    if not cached_ds_path.exists():
      img_files, mask_files = self._get_fmriprep_files()
      time_series = self.extract_timeseries(img_files, mask_files, self.atlas_name)
      dataset = self.create_dataset(time_series)
      self.cache_dataset(dataset, overwrite=False)
    else:
      dataset = xr.open_dataset(cached_ds_path, engine='h5netcdf')
      self.feature_names_ = dataset.coords['region'].values.tolist()

    return self

  def transform(self, X=None):  # noqa: N803
    """Extract region time-series from the dataset.

    Parameters
    ----------
    X : list, optional
        list of participant ids to select (BIDS-compatible without `sub-` prefix), e.g.,
        ['AVGP-01', 'NVGP-03'] returns two subjects. Passing `None` will return all
        the subjects. Defaults to None.

    Returns
    -------
    xarray.Dataset
        dataset with the following variables:
          - subject: (len(X))
          - timeseries: (n_subjects, n_timepoints, n_regions)

    Raises
    ------
    ValueError
        Parcellation has not been fitted yet. Call fit() before calling transform().
    """

    # load from cache
    if self.cache_dir:
      cached_ds_path = Path(self.cache_dir).expanduser() / f'timeseries_{self.atlas_name}.nc5'
      if cached_ds_path.exists():
        dataset = xr.open_dataset(cached_ds_path, engine='h5netcdf')
        dataset = dataset.set_coords('region')

    if X is not None:
      selected_subjects = X.reshape(-1).tolist()
      dataset = dataset.sel(subject=selected_subjects)

    if self.normalize:
      dataset['timeseries'] = self._normalize_func(dataset['timeseries'])

    return dataset

  def get_feature_names_out(self, input_features):
    return self.feature_names_
