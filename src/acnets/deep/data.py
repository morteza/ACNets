import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

from sklearn.preprocessing import LabelEncoder
from ..pipeline import Parcellation, ConnectivityExtractor, TimeseriesAggregator, ConnectivityAggregator


class ACNetsDataModule(pl.LightningDataModule):
    def __init__(self,
                    atlas='dosenbach2010',
                    kind='partial correlation',
                    test_ratio=.25, val_ratio=.125,
                    suffle=True, batch_size=8):
        super().__init__()
        self.atlas = atlas
        self.kind = kind
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = 1 - test_ratio - val_ratio
        self.shuffle = suffle
        self.batch_size = batch_size
        self.y_encoder = LabelEncoder()


    def prepare_data(self):
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

        # TODO extract subject labels (AVGP or NVGP)
        y = self.y_encoder.fit_transform([s[:4] for s in h1_time_regions['subject'].values])
        y = torch.tensor(y)

        data = TensorDataset(h1, h2, h3, h4, h5, y)
        n_subjects = len(data)
        self.train, self.val, self.test = random_split(
            data,
            [int(n_subjects * self.train_ratio),
                int(n_subjects * self.val_ratio),
                int(n_subjects * self.test_ratio)])

        # TODO concat val to train
        self.train = ConcatDataset([self.train, self.val])

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=self.shuffle)


    # TODO WIP: reshape as tidy dataframes

    # h1
    # h1 = h1_time_regions['timeseries'].to_dataframe().reset_index().pivot(index=['subject', 'region'], columns='timepoint')
    # h1.columns = h1.columns.droplevel(0)

    # h3
    # h3 = h3_time_networks['timeseries'].to_dataframe().reset_index().pivot(index=['subject', 'network'], columns='timepoint')
    # h3.columns = h3.columns.droplevel(0)
