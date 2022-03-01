import re
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from pathlib import Path
from nilearn.interfaces.bids import get_bids_files
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import datasets as nilearn_datasets
from nilearn import maskers
from tqdm import tqdm


class Parcellation(TransformerMixin, BaseEstimator):
  """# to parcellate activities given a parcellation atlas.
  """

  def __init__(self,
               bids_dir=Path('data/julia2018'),
               atlas_name='cort-maxprob-thr25-2mm',
               denoise_strategy='simple',
               fmriprep_bids_space='MNI152NLin2009cAsym',
               cache_folder=Path('data/julia2018_resting'),
               verbose=0) -> None:

    self.dataset: xr.Dataset = None

    self.verbose = verbose
    self.atlas_name = atlas_name
    self.denoise_strategy = denoise_strategy
    self.cache_folder = Path(cache_folder)
    self.bids_dir = Path(bids_dir)

    self.fmriprep_dir = self.bids_dir / 'derivatives/fmriprep'
    self.fmriprep_bids_space = fmriprep_bids_space

    # load from cache or look for fmriprep derivatives
    cached_dataset_path = self.cache_folder / f'timeseries_{self.atlas_name}.nc'
    if cached_dataset_path.exists():
      self.dataset = xr.open_dataset(cached_dataset_path)
    elif not self.fmriprep_dir.exists():
        raise ValueError(f'{self.fmriprep_dir} does not exist')

    super().__init__()

  def get_fmriprep_files(self):
    # RESTING SCANS
    img_files = get_bids_files(
        self.fmriprep_dir,
        file_tag='bold',
        modality_folder='func',
        filters=[('ses', 'rest'),
                 ('space', self.fmriprep_bids_space),
                 ('desc', 'preproc')],
        file_type='nii.gz')

    # BRAIN MASKS
    mask_files = get_bids_files(
        self.fmriprep_dir,
        file_tag='mask',
        modality_folder='func',
        filters=[('ses', 'rest'),
                 ('space', self.fmriprep_bids_space),
                 ('desc', 'brain')],
        file_type='nii.gz')

    return img_files, mask_files

  def load_masker(self, atlas_name, mask_img, return_atlas_labels=True):
    if 'cort-maxprob' in atlas_name:
      atlas = nilearn_datasets.fetch_atlas_harvard_oxford(atlas_name)
      atlas_labels_img = atlas.maps

      masker = maskers.NiftiLabelsMasker(
          labels_img=atlas_labels_img,
          mask_img=mask_img,
          standardize=True,
          #  memory='tmp/nilearn_cache',
          verbose=0)

      if return_atlas_labels:
        atlas_labels = pd.DataFrame(atlas.labels, columns=['region'])
        atlas_labels = atlas_labels.drop(index=[0]).set_index('region')
        return masker, atlas_labels
      else:
        return masker

    raise ValueError('Atlas name {} not recognized.'.format(atlas_name))

  def extract_timeseries(self, img_files, mask_files, atlas_name):

    time_series = {}

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
      masker, atlas_labels = self.load_masker(atlas_name, mask, return_atlas_labels=True)
      ts = masker.fit_transform(img, confound, sample_mask)

      if ts.shape[0] > len(valid_timepoints):
        ts = ts[timepoints_mask]

      time_series[subject] = ts

    return time_series, atlas_labels

  def create_dataset(self, atlas_labels, time_series=None):
    
    # dims: (n_subjects, n_timepoints, n_regions)
    _time_series_arr = np.stack(list(time_series.values()))
    preprocessed_subjects = list(time_series.keys())

    # atlas dataset
    atlas_ds = atlas_labels.to_xarray()

    # time-series dataset
    time_series_ds = xr.Dataset({
        'timeseries': (['subject', 'timepoint', 'region'], _time_series_arr)
    }, coords={
        'timepoint': np.arange(1, _time_series_arr.shape[1] + 1),
        'region': atlas_labels.index,
        'subject': preprocessed_subjects,
    })

    # participants dataset
    bids_participants = pd.read_csv('data/julia2018/participants.tsv', sep='\t')
    bids_participants.rename(columns={'participant_id': 'subject'}, inplace=True)
    # remove "sub-" prefix
    bids_participants['subject'] = bids_participants['subject'].apply(lambda x: x.split('-')[1])
    bids_participants.set_index('subject', inplace=True)
    bids_participants = bids_participants.query('index in @preprocessed_subjects')
    participants_ds = bids_participants.to_xarray()

    # merge arrays into a single dataset
    _ds = xr.merge([time_series_ds, participants_ds, atlas_ds])

    return _ds

  def get_tr(self):
    """ use pybids to extract TR from BIDS metadata.

    Returns
    -------
    int
        Repetition time in seconds
    """

    from bids import BIDSLayout
    layout = BIDSLayout(self.fmriprep_dir, derivatives=True)
    t_r = layout.get_tr(derivatives=True, task='rest')
    return t_r

  def save_to_cache(self, overwrite=False):
    if not self.dataset:
      raise ValueError('Parcellation has not been fitted yet.')

    cached_ds_path = self.cache_folder / f'timeseries_{self.atlas_name}.nc'

    if overwrite or not cached_ds_path.exists():
      self.dataset.to_netcdf(cached_ds_path, engine='netcdf4')

  def fit(self, X=None, y=None, **fit_params):  # noqa: N803

    if not self.dataset:
      img_files, mask_files = self.get_fmriprep_files()
      time_series, atlas_labels = self.extract_timeseries(img_files, mask_files, self.atlas_name)
      self.dataset = self.create_dataset(atlas_labels, time_series)

    return self

  def transform(self, X):  # noqa: N803
    if not self.dataset:
      raise ValueError('Parcellation has not been fitted yet.')

    raise NotImplementedError()
