from bids import BIDSLayout
from nilearn import datasets, plotting, maskers
from collections import namedtuple
import re
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


BIDSDataset = namedtuple(
    'BIDSDataset',
    ['images', 'mask_images', 'confounds_files', 't_r']
)


def load_bids_dataset(root):
  layout = BIDSLayout(
      root,
      derivatives=True,
      database_path='tmp/pybids_cache/julia2018')

  # all_subjects = layout.get_subject()
  t_r = layout.get_tr(task='rest')

  images = layout.get(
      task='rest',
      desc='preproc',
      suffix='bold',
      scope='fmriprep',
      extension='nii.gz',
      return_type='filename')

  mask_images = layout.get(
      task='rest',
      desc='brain',
      suffix='mask',
      scope='fmriprep',
      extension='nii.gz',
      return_type='filename')

  confounds_files = layout.get(
      task='rest',
      desc='confounds',
      suffix='timeseries',
      scope='fmriprep',
      extension='tsv',
      return_type='filename')

  return BIDSDataset(images, mask_images, confounds_files, t_r)


def extract_difumo_timeseries(
    dimension: int = 64,
    resolution_mm: int = 2,
    confounds_cols=[
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'csf', 'global_signal', 'a_comp_cor_00', 'a_comp_cor_01'],
    root: Path = Path('data/julia2018/')
):
    
  assert root.exists(), '{} does not exist.'.format(root.absolute())

  # atlas
  atlas = datasets.fetch_atlas_difumo(
      dimension=dimension,
      resolution_mm=resolution_mm,
      legacy_format=False)

  atlas_coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas.maps)
  atlas_regions = pd.concat([atlas.labels, pd.DataFrame(atlas_coordinates)], axis=1)
  atlas_regions.rename(columns={0: 'mni152_x',
                                1: 'mni152_y',
                                2: 'mni152_z'}, inplace=True)

  atlas_regions.drop(columns=['component'], inplace=True)
  atlas_regions.index.name = 'region'

  # load rs-fMRI session from the BIDS dataset
  bids_ds = load_bids_dataset(root)

  timeseries = {}
  subjects = []

  bids_ds_iter = tqdm(zip(
      bids_ds.images,
      bids_ds.mask_images,
      bids_ds.confounds_files), total=len(bids_ds.images))

  for img, mask_img, confound_files in bids_ds_iter:
    subject = re.search('func/sub-(.*)_ses', img)[1]
    subjects.append(subject)

    tmp_root = Path(f'tmp/difumo{dimension}_{resolution_mm}_cache/')
    tmp_root.mkdir(parents=True, exist_ok=True)

    tmp_ts_path = tmp_root / f'{subject}_timeseries.npz'

    if tmp_ts_path.exists():
    #   print(f'Loading {subject} from cache...')
      ts = np.load(tmp_ts_path)['arr_0']
      timeseries[subject] = ts
      continue

    masker = maskers.NiftiMapsMasker(
        atlas.maps,
        mask_img=mask_img,
        detrend=True,
        standardize=True,
        t_r=bids_ds.t_r,
        verbose=0)
    # memory='tmp/nilearn_cache', memory_level=1,

    confounds = pd.read_csv(confound_files, sep='\t', usecols=confounds_cols).values
    ts = masker.fit_transform(img, confounds=confounds)

    np.savez(tmp_ts_path, ts)
    timeseries[subject] = ts

  # reshape dim-0 is subject, dim-1 is region, dim-2 is time point
  timeseries = np.array(list(timeseries.values())).transpose(0, 2, 1)

  ds = xr.Dataset({
      'timeseries': (['subject', 'region', 'timestep'], timeseries)
  })

  # subject data
  bids_participants = pd.read_csv(root / 'participants.tsv', sep='\t')
  bids_participants.rename(columns={'participant_id': 'subject'}, inplace=True)
  bids_participants['subject'] = bids_participants['subject'].apply(
      lambda x: x.split('-')[1])
  bids_participants.set_index('subject', inplace=True)
  bids_participants = bids_participants.query('index in @subjects')

  # metadata
  ds.coords['timestep'] = range(0, 125)
  ds.attrs['description'] = 'Resting-state time series from fmriprep scans based on DIFUMO atlas'
  ds.coords['subject'].attrs['description'] = (
      'subject identifier; AVGPxx for action video'
      'gamers and NVGPxx for non-video gamers')

  # merge all datasets and store
  ds = xr.merge([ds, bids_participants.to_xarray(), atlas_regions.to_xarray()])
  return ds
