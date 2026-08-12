from torch import nn

from ._fine_tunable import FineTunable
from .mvae import MaskedVAE
from .seq2seq import Seq2SeqAE


class Classifier(nn.Module):
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
    """Autoencoder over masked input, with a classification head.

    Args:
        x (torch.Tensor): shape: (subjects, timepoints, regions)

    Returns:
        y, h, loss_recon
    """

    def __init__(self, n_features, n_embeddings, input_type):
        super().__init__()

        self.save_hyperparameters(n_features=n_features,
                                  n_embeddings=n_embeddings,
                                  input_type=input_type)
        self.model_name = f'masked_{input_type}'

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

        h, _x_recon, loss_recon = self.fe(x)
        # FIXME to reuse x_recon, it should be Unflattened

        y = self.cls_head(h) if self.phase == 'finetune' else None

        return y, h, loss_recon
