import os
from typing import Literal
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from ..pipeline import Parcellation, ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator
from ..parcellations import dosenbach
from sklearn.model_selection import train_test_split
from pathlib import Path


class Julia2018DataModule(pl.LightningDataModule):
    """Julia2018 preprocessed resting-state dataset.

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
            y: subject labels (AVGP or NVGP)

    """

    def __init__(self,
                 atlas='dosenbach2010',
                 kind='partial correlation',
                 aggregation_strategy: Literal[
                     'time_regions', 'time_networks', 'time_wavelets',
                     'conn_regions', 'cconn_networks', 'tconn_networks'] = 'time_regions',
                 dataset_path=Path('data/julia2018/'),
                 segment_length: int = -1,  # -1 for full timeseries
                 test_ratio=.25,
                 shuffle=True,
                 batch_size=8,
                 num_workers=os.cpu_count() - 1):  # type: ignore

        super().__init__()
        self.save_hyperparameters()

        self.y_encoder = LabelEncoder()

        match atlas:
            case 'dosenbach2010':
                self._atlas_masker, self._parcels = dosenbach.load_dosenbach2010_masker()
            case 'aal':
                from nilearn import maskers
                import pandas as pd

                self._parcels = pd.read_csv('data/atlases/AAL3v1.csv')
                self._parcels.dropna(subset=['index'], inplace=True)
                self._parcels.set_index('region', inplace=True)

                self._atlas_masker = maskers.NiftiLabelsMasker(
                    'data/atlases/AAL3v1.nii.gz',
                    standardize='zscore_sample',
                    standardize_confounds='zscore_sample',
                    # detrend=True,
                    verbose=0)

            case _:
                raise ValueError(f'Atlas {atlas} is not supported.')

    def prepare_data(self):
        # just calling parcellation once, so time-series will be cached
        self.get_timeseries(self.hparams['dataset_path'])

    def get_timeseries(self, dataset_path, n_subjects=None):
        x_time_regions = Parcellation(
            atlas_name=self.hparams['atlas'],
            bids_dir=dataset_path,
            fmriprep_bids_space='MNI152NLin2009cAsym',
            normalize=True
        ).fit_transform(X=None)
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

        x_time_regions = self.get_timeseries(self.hparams['dataset_path'])

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

        # extract subject labels (AVGP or NVGP)
        y = self.y_encoder.fit_transform([s[:4] for s in x_time_regions['subject'].values])
        y = torch.tensor(y)

        if 'time' in self.hparams['aggregation_strategy'] and self.hparams['segment_length'] > 0:
            x = self.segment_timeseries(x)
            # TODO match y to x
            y = y.repeat_interleave(x.shape[0] // y.shape[0])

        self.full_data = TensorDataset(x, y)

        # stratified split into train, val and test
        train_idx, test_idx = train_test_split(
            torch.arange(len(y)),
            test_size=self.hparams['test_ratio'], stratify=y, shuffle=self.hparams['shuffle'])

        self.train = torch.utils.data.Subset(self.full_data, train_idx)
        self.test = torch.utils.data.Subset(self.full_data, test_idx)

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
