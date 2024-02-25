from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from .cvae import CVAE
from ._fine_tunable import FineTunable


class Classifier(FineTunable):
    def __init__(self, n_inputs=32):
        super().__init__()

        self.n_inputs = n_inputs

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.model = nn.Sequential(
            nn.Linear(n_inputs, n_inputs),
            nn.ReLU(),
            nn.Linear(n_inputs, n_inputs // 2),
            nn.ReLU(),
            nn.Linear(n_inputs // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class WaveletModel(pl.LightningModule):
    def __init__(self, n_regions, n_wavelets=124, n_embeddings=2, segment_length=32):
        super().__init__()

        self.save_hyperparameters()

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        self.feature_extractor = CVAE(n_regions, n_embeddings, kernel_size=5, stride=2)
        self.cls_head = Classifier(n_embeddings)

    def forward(self, x_wvt):
        """_summary_

        Args:
            x_wvt (torch.Tensor): shape: (subjects, wavelets, regions)

        Returns:
            y, h, loss_recon
        """

        x = x_wvt[:, :self.n_wavelets, :]  # -> shape: (subjects, wavelets, regions)

        x_segments = x.unfold(1, self.segment_length, self.segment_length)
        x_segments = x_segments.reshape(-1, self.segment_length, self.n_regions)
        x_segments = x_segments.permute(0, 2, 1)  # -> shape: (subjects * segments, regions, segment_length)

        h, x_segments_recon, loss_recon = self.feature_extractor(x_segments)

        if self.phase == 'finetune':
            y = self.cls_head(h)
            y = y.reshape(x.size(0), -1, 2)
            y = y.mean(dim=1)
        else:
            y = None

        return y, h, loss_recon
