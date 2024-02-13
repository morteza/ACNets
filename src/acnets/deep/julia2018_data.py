import os
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

from sklearn.preprocessing import LabelEncoder
from ..pipeline import Parcellation, ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator
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
                 dataset_path=Path('data/julia2018/'),
                 test_ratio=.25,
                 shuffle=True,
                 batch_size=8,
                 num_workers=os.cpu_count() - 1):

        super().__init__()
        self.atlas = atlas
        self.kind = kind
        self.dataset_path = dataset_path
        self.test_ratio = test_ratio
        self.train_ratio = 1 - test_ratio
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.y_encoder = LabelEncoder()
        self.num_workers = num_workers

    def prepare_data(self):
        # just calling parcellation once, so time-series will be cached
        Parcellation(
            atlas_name=self.atlas,
            bids_dir=self.dataset_path,
            fmriprep_bids_space='MNI152NLin2009cAsym',
            normalize=True
        ).fit_transform(X=None)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            x1_time_regions = Parcellation(
                atlas_name=self.atlas,
                bids_dir=self.dataset_path,
                fmriprep_bids_space='MNI152NLin2009cAsym',
                normalize=True
            ).fit_transform(X=None)
            x2_conn_regions = ConnectivityExtractor(kind=self.kind).fit_transform(x1_time_regions)
            x3_time_networks = TimeseriesAggregator(strategy='network').fit_transform(x1_time_regions)
            x4_conn_networks = ConnectivityExtractor(kind=self.kind).fit_transform(x3_time_networks)
            x5_conn_networks = ConnectivityAggregator(strategy='network').fit_transform(x2_conn_regions)
            x6_time_wavelets = TimeseriesAggregator(strategy='wavelet',
                                                    wavelet_name='db1').fit_transform(x1_time_regions)

            x1 = torch.Tensor(x1_time_regions['timeseries'].values)
            x2 = torch.Tensor(x2_conn_regions['connectivity'].values)
            x3 = torch.Tensor(x3_time_networks['timeseries'].values)
            x4 = torch.Tensor(x4_conn_networks['connectivity'].values)
            x5 = torch.Tensor(x5_conn_networks['connectivity'].values)
            x6 = torch.Tensor(x6_time_wavelets['wavelets'].values)

            # extract subject labels (AVGP or NVGP)
            y = self.y_encoder.fit_transform([s[:4] for s in x1_time_regions['subject'].values])
            y = torch.tensor(y)

            self.full_data = TensorDataset(x1, x2, x3, x4, x5, x6, y)

            # stratified split into train, val and test
            n_subjects = len(y)
            train_idx, test_idx = train_test_split(
                torch.arange(n_subjects), test_size=self.test_ratio, stratify=y, shuffle=self.shuffle)

            self.train = torch.utils.data.Subset(self.full_data, train_idx)
            self.test = torch.utils.data.Subset(self.full_data, test_idx)

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
