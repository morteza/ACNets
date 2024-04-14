import os
from typing import Literal
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from ..preprocessing import ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator
from sklearn.model_selection import train_test_split
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
        test_ratio (float): default=.25
            The ratio of the dataset to include in the test split.

    Attributes:
        All the train, val, test and full_data attributes are torch.utils.data.Dataset objects
        and contain the following attributes:
            time_regions: (subjects, timepoints, regions)
            time_networks: (subjects, timepoints, networks)
            time_wavelets: (subjects, timepoints, wavelets)
            conn_regions: (subjects, regions, regions)
            cconn_networks: (subjects, networks, networks)
            tconn_networks: (subjects, networks, networks)
    """

    def __init__(self,
                 atlas='dosenbach2010',
                 kind='partial correlation',
                 aggregation_strategy: Literal[
                     'time_regions', 'time_networks', 'time_wavelets',
                     'conn_regions', 'cconn_networks', 'tconn_networks'] = 'time_regions',
                 dataset_path=Path('/mnt/Lifestream/MPI-LEMON/MRI_Preprocessed_Derivatives/'),
                 n_subjects: int = 5,
                 segment_length: int = -1,  # -1 for full timeseries
                 test_ratio=.25,
                 val_ratio=.125,
                 normalize=True,
                 shuffle=True,
                 batch_size=8,
                 num_workers=os.cpu_count() - 1):  # noqa

        super().__init__()
        self.save_hyperparameters()

        self.y_encoder = LabelEncoder()
        self.timeseries_dataset_path = Path(f'data/LEMON/timeseries_{atlas}.nc')

        match atlas:
            case 'dosenbach2010':
                pass
            case 'aal':
                pass
            case _:
                raise ValueError(f'Atlas {atlas} is not supported.')

    def normalize_timeseries(self, x: xr.DataArray):
        """Normalize the subject data to [-1, 1] range."""
        x_min, x_max = x.min(['region', 'timepoint']), x.max(['region', 'timepoint'])
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = x_norm * 2 - 1  # map 0 to -1, and 1 to 1
        return x_norm

    def prepare_data(self):

        # if there is a dataset file, check if the number of subjects is enough
        if self.timeseries_dataset_path.exists():
            with xr.open_dataset(self.timeseries_dataset_path) as dataset:
                dataset.load()
            n_available_subjects = dataset.sizes['subject']
            if n_available_subjects >= self.hparams['n_subjects']:
                return
        else:
            n_available_subjects = 0
            dataset = xr.Dataset()
            dataset.attrs['space'] = 'MNI2mm'

        print(self.timeseries_dataset_path)

    def get_timeseries(self, dataset_path, n_subjects):
        with xr.open_dataset(dataset_path, engine='h5netcdf') as dataset:
            dataset.load()
            x_time_regions = dataset.isel(subject=slice(n_subjects))
        return x_time_regions

    def segment_timeseries(self, x):
        if self.hparams['segment_length'] > 0:
            n_features = x.shape[-1]
            x = x.unfold(1, self.hparams['segment_length'], self.hparams['segment_length'])
            x = x.reshape(-1, self.hparams['segment_length'], n_features)
        return x

    def setup(self, stage=None):

        if stage != 'fit' and stage is not None:
            # skip setup if reloading is not required
            return

        t2_mni2mm_files = sorted(self.hparams['dataset_path'].glob('**/func/*MNI2mm.nii.gz'))

        if len(t2_mni2mm_files) > 0:
            self.hparams['n_subjects'] = min(self.hparams['n_subjects'], len(t2_mni2mm_files))

        # time-series (index = 0)
        x_time_regions = self.get_timeseries(self.timeseries_dataset_path, self.hparams['n_subjects'])

        match self.hparams['aggregation_strategy']:
            case 'time_regions':
                x = torch.Tensor(x_time_regions['timeseries'].values)
            case 'conn_regions':
                x_conn_regions = ConnectivityExtractor(
                    kind=self.hparams['kind']).fit_transform(x_time_regions)
                x = torch.Tensor(x_conn_regions['connectivity'].values)
            case 'time_wavelets':
                x_time_wavelets = TimeseriesAggregator(
                    strategy='wavelet', wavelet_name='db1').fit_transform(x_time_regions)
                x = torch.Tensor(x_time_wavelets['wavelets'].values)
            case 'time_networks' if 'network' in x_time_regions.data_vars.keys():
                x_time_networks = TimeseriesAggregator(strategy='network').fit_transform(x_time_regions)
                x = torch.Tensor(x_time_networks['timeseries'].values)
            case 'tconn_networks' if 'network' in x_time_regions.data_vars.keys():
                x_time_networks = TimeseriesAggregator(strategy='network').fit_transform(x_time_regions)
                x = torch.Tensor(x_time_networks['timeseries'].values)
                x_tconn_networks = ConnectivityExtractor(
                    kind=self.hparams['kind']).fit_transform(x_time_networks)
                x = torch.Tensor(x_tconn_networks['connectivity'].values)
            case 'cconn_networks' if 'network' in x_time_regions.data_vars.keys():
                x_conn_regions = ConnectivityExtractor(
                    kind=self.hparams['kind']).fit_transform(x_time_regions)
                x_cconn_networks = ConnectivityAggregator(strategy='network').fit_transform(x_conn_regions)
                x = torch.Tensor(x_cconn_networks['connectivity'].values)
            case _:
                raise ValueError(f'Aggregation strategy {self.hparams["aggregation_strategy"]} '
                                 'is not supported.')

        if 'time' in self.hparams['aggregation_strategy'] and self.hparams['segment_length'] > 0:
            x = self.segment_timeseries(x)

        self.full_data = TensorDataset(x)

        # split subjects into train, val and test
        if self.hparams['test_ratio'] is not None:
            train_idx, test_idx = train_test_split(
                torch.arange(x.shape[0]), test_size=self.hparams['test_ratio'],
                shuffle=self.hparams['shuffle'])

            self.train = torch.utils.data.Subset(self.full_data, train_idx)
            self.test = torch.utils.data.Subset(self.full_data, test_idx)

            # TODO separate val dataset
            # concat val and train for final training (we only use train and test for now)
            # train_idx, val_idx = train_test_split(
            #        train_idx, test_size=self.val_ratio / (1 - self.test_ratio),
            #        shuffle=self.shuffle)
            # self.val = torch.utils.data.Subset(self.full_data, val_idx)
            # self.train = ConcatDataset([self.train, self.val])

        else:
            self.train = self.full_data
            self.test = self.full_data

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          persistent_workers=True)

    def val_dataloader(self):
        # FIXME Note this is the same as the test dataloader (change to self.val for separate validation set)
        return DataLoader(self.test, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          persistent_workers=True)
