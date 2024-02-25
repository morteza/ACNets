from ._fine_tunable import FineTunable
from torch import nn
import pytorch_lightning as pl
import torchmetrics as metrics
from .mvae import MaskedVAE
from .seq2seq import Seq2SeqAE


class Classifier(pl.LightningModule):
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


class MaskedModel(FineTunable):
    def __init__(self, n_regions, n_embeddings=2):
        super().__init__()

        self.save_hyperparameters()

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        # self.feature_extractor = MaskedVAE(segment_length, n_embeddings, mask_size=4)
        self.feature_extractor = Seq2SeqAE(n_regions, n_embeddings, mask_size=4)
        self.cls_head = Classifier(n_inputs=n_embeddings)

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): shape: (subjects, timepoints, regions)

        Returns:
            y, h, loss_recon
        """

        h, x_recon, loss_recon = self.feature_extractor(x)

        if self.phase == 'finetune':
            y = self.cls_head(h)
        else:
            y = None

        return y, h, loss_recon
