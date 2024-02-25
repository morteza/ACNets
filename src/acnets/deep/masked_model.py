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
    """_summary_

    Args:
        x (torch.Tensor): shape: (subjects, timepoints, regions)

    Returns:
        y, h, loss_recon
    """

    def __init__(self, n_features, n_embeddings, input_type):
        super().__init__()

        self.save_hyperparameters()
        self.model_name = f'masked_{input_type}'

        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=2)

        if 'time' in input_type:
            self.fe = Seq2SeqAE(n_features, n_embeddings)
            # self.fe = MaskedVAE(segment_length, n_embeddings, mask_size=4)
        elif 'conn' in input_type:
            self.fe = nn.Sequential(
                nn.Flatten(),
                MaskedVAE(n_features, n_embeddings, mask_size=0)
            )

        self.cls_head = Classifier(n_inputs=n_embeddings)

    def forward(self, x):

        h, x_recon, loss_recon = self.fe(x)
        # FIXME to reuse x_recon, it should be Unflattened

        if self.phase == 'finetune':
            y = self.cls_head(h)
        else:
            y = None

        return y, h, loss_recon
