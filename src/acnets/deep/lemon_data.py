import os
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

from sklearn.preprocessing import LabelEncoder
from ..pipeline import Parcellation, ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator
from sklearn.model_selection import train_test_split
from src.acnets.parcellations.dosenbach import load_dosenbach2010_masker
from pathlib import Path
from joblib import Parallel, delayed
import xarray as xr
import numpy as np


class LEMONDataModule(pl.LightningDataModule):
    """MPI-LEMON dataset for multi-modal brain connectivity analysis.

    Args:
        atlas (str): default='dosenbach2010'
            The name of the atlas to use for parcellation.
        kind (str) default='partial correlation'
            The kind of connectivity to extract.
        test_ratio float): default=.25
            The ratio of the dataset to include in the test split.

    Attributes:
        All the train, val, test and full_data attributes are torch.utils.data.Dataset objects
        and contain the following attributes:
            x1: time-series (regions x timepoints)
            x2: connectivity (regions x regions)
            x3: time-series (networks x timepoints)
            x4: connectivity (networks x networks)
            x5: connectivity (networks x networks)

    """

    def __init__(self,
                 atlas='dosenbach2010',
                 kind='partial correlation',
                 dataset_path=Path('/mnt/Lifestream/MPI-LEMON/MRI_Preprocessed_Derivatives/'),
                 n_subjects=5,
                 test_ratio=.25,
                 val_ratio=.125,
                 shuffle=True,
                 batch_size=8,
                 num_workers=os.cpu_count() - 1):

        # TODO support other atlases
        if atlas != 'dosenbach2010':
            raise ValueError('Only dosenbach2010 atlas is supported for now.')

        super().__init__()
        self.atlas = atlas
        self.kind = kind
        self.dataset_path = dataset_path
        self.n_subjects = n_subjects
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = 1 - test_ratio - val_ratio
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.y_encoder = LabelEncoder()
        self.num_workers = num_workers

        self.timeseries_dataset_path = Path(f'data/mpi-lemon/{self.n_subjects}subjects_{self.atlas}_timeseries.nc')

    def extract_timeseries(self, t2_mni_file):
        subject = t2_mni_file.parents[1].stem
        atlas_masker, _ = load_dosenbach2010_masker()
        ts = atlas_masker.fit_transform(t2_mni_file)  # (m_timepoints, n_regions)
        return subject, ts

    def normalize_timeseries(self, x: xr.DataArray):
        """Normalize the subject data to [-1, 1] range."""
        x_norm = (x - x.min(['subject'])) / (x.max(['subject']) - x.min(['subject']))
        x_norm = x_norm * 2 - 1  # map 0 to -1, and 1 to 1
        return x_norm

    def prepare_data(self):

        if self.timeseries_dataset_path.exists():
            # skip preprocessing if the dataset file already exists
            print('skip preprocessing')
            return

        t2_mni2mm_files = sorted(self.dataset_path.glob('**/func/*MNI2mm.nii.gz'))

        timeseries = Parallel(n_jobs=self.num_workers)(
            delayed(self.extract_timeseries)(t2_mni_file)
            for t2_mni_file in t2_mni2mm_files[:self.n_subjects])
        timeseries = dict(timeseries)

        _, regions = load_dosenbach2010_masker()
        regions = regions.to_xarray().drop_vars('index')

        dataset = xr.Dataset()
        dataset.attrs['space'] = 'MNI2mm'
        dataset['timeseries'] = xr.DataArray(
            np.stack(list(timeseries.values())),
            dims=('subject', 'timepoint', 'region'),
            coords={'subject': list(timeseries.keys())}
        )

        dataset = xr.merge([dataset, regions])

        dataset['timeseries'] = self.normalize_timeseries(dataset['timeseries'])

        dataset.to_netcdf(self.timeseries_dataset_path, engine='h5netcdf')

    def setup(self, stage=None):

        if stage != 'fit' and stage is not None:
            # skip setup if reloading is not required
            return

        if not self.timeseries_dataset_path.exists():
            # run preprocessing if the dataset file does not exist
            self.prepare_data()

        x1_time_regions = xr.open_dataset(self.timeseries_dataset_path, engine='h5netcdf')

        x2_conn_regions = ConnectivityExtractor(kind=self.kind).fit_transform(x1_time_regions)
        x3_time_networks = TimeseriesAggregator(strategy='network').fit_transform(x1_time_regions)
        x4_conn_networks = ConnectivityExtractor(kind=self.kind).fit_transform(x3_time_networks)
        x5_conn_networks = ConnectivityAggregator(strategy='network').fit_transform(x2_conn_regions)

        x1 = torch.Tensor(x1_time_regions['timeseries'].values)
        x2 = torch.Tensor(x2_conn_regions['connectivity'].values)
        x3 = torch.Tensor(x3_time_networks['timeseries'].values)
        x4 = torch.Tensor(x4_conn_networks['connectivity'].values)
        x5 = torch.Tensor(x5_conn_networks['connectivity'].values)

        self.full_data = TensorDataset(x1, x2, x3, x4, x5)

        # split into train, val and test
        n_subjects = x1.shape[0]
        train_idx, test_idx = train_test_split(
            torch.arange(n_subjects), test_size=self.test_ratio, shuffle=self.shuffle)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=self.val_ratio / (1 - self.test_ratio),
            shuffle=self.shuffle)

        self.train = torch.utils.data.Subset(self.full_data, train_idx)
        self.val = torch.utils.data.Subset(self.full_data, val_idx)
        self.test = torch.utils.data.Subset(self.full_data, test_idx)

        # concat val and train for final training (we only use train and test for now)
        self.train = ConcatDataset([self.train, self.val])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        # FIXME Note this is the same as the test dataloader (change to self.val for separate validation set)
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)
