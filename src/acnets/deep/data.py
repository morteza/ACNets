import os
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

from sklearn.preprocessing import LabelEncoder
from ..pipeline import Parcellation, ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator
from sklearn.model_selection import train_test_split


class ACNetsDataModule(pl.LightningDataModule):
    """_summary_

    Data
    ----
    - h1: time-series (regions x timepoints)
    - h2: connectivity (regions x regions)
    - h3: time-series (networks x timepoints)
    - h4: connectivity (networks x networks)
    - h5: connectivity (networks x networks)


    """
    def __init__(self,
                 atlas='dosenbach2010',
                 kind='partial correlation',
                 test_ratio=.25, val_ratio=.125,
                 suffle=True, batch_size=8, num_workers=os.cpu_count() - 1):
        super().__init__()
        self.atlas = atlas
        self.kind = kind
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = 1 - test_ratio - val_ratio
        self.shuffle = suffle
        self.batch_size = batch_size
        self.y_encoder = LabelEncoder()
        self.num_workers = num_workers

    def prepare_data(self):
        # just calling parcellation once, so time-series will be cached
        Parcellation(atlas_name=self.atlas).fit_transform(X=None)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            h1_time_regions = Parcellation(atlas_name=self.atlas).fit_transform(X=None)
            h2_conn_regions = ConnectivityExtractor(kind=self.kind).fit_transform(h1_time_regions)
            h3_time_networks = TimeseriesAggregator(strategy='network').fit_transform(h1_time_regions)
            h4_conn_networks = ConnectivityExtractor(kind=self.kind).fit_transform(h3_time_networks)
            h5_conn_networks = ConnectivityAggregator(strategy='network').fit_transform(h2_conn_regions)
            h1 = torch.Tensor(h1_time_regions['timeseries'].values)
            h2 = torch.Tensor(h2_conn_regions['connectivity'].values)
            h3 = torch.Tensor(h3_time_networks['timeseries'].values)
            h4 = torch.Tensor(h4_conn_networks['connectivity'].values)
            h5 = torch.Tensor(h5_conn_networks['connectivity'].values)

            # extract subject labels (AVGP or NVGP)
            y = self.y_encoder.fit_transform([s[:4] for s in h1_time_regions['subject'].values])
            y = torch.tensor(y)

            self.full_data = TensorDataset(h1, h2, h3, h4, h5, y)

            # stratified split into train, val and test
            n_subjects = len(y)
            train_idx, test_idx = train_test_split(
                torch.arange(n_subjects), test_size=self.test_ratio, stratify=y, shuffle=self.shuffle)
            train_idx, val_idx = train_test_split(
                train_idx, test_size=self.val_ratio / (1 - self.test_ratio),
                stratify=y[train_idx], shuffle=self.shuffle)

            self.train = torch.utils.data.Subset(self.full_data, train_idx)
            self.val = torch.utils.data.Subset(self.full_data, val_idx)
            self.test = torch.utils.data.Subset(self.full_data, test_idx)

            # concat val and train for final training (we only use train and test for now)
            self.train = ConcatDataset([self.train, self.val])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        # Note this is the same as the test dataloader (change to self.val for separate validation set)
        return DataLoader(self.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    # TODO WIP: reshape as tidy dataframes

    # h1
    # h1 = h1_time_regions['timeseries'].to_dataframe().reset_index().pivot(index=['subject', 'region'], columns='timepoint')
    # h1.columns = h1.columns.droplevel(0)

    # h3
    # h3 = h3_time_networks['timeseries'].to_dataframe().reset_index().pivot(index=['subject', 'network'], columns='timepoint')
    # h3.columns = h3.columns.droplevel(0)
