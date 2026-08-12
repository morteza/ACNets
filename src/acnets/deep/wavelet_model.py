from torch import nn

from ._fine_tunable import FineTunable
from .cvae import CVAE


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


class WaveletModel(FineTunable):
    def __init__(self, n_regions, n_wavelets=124, n_embeddings=2, segment_length=32):
        super().__init__()

        self.save_hyperparameters(n_regions=n_regions, n_wavelets=n_wavelets,
                                  n_embeddings=n_embeddings,
                                  segment_length=segment_length)
        self.n_regions = n_regions
        self.n_wavelets = n_wavelets
        self.segment_length = segment_length

        self.feature_extractor = CVAE(n_regions, n_embeddings, kernel_size=5, stride=2)
        self.cls_head = Classifier(n_embeddings)

    def forward(self, x_wvt):
        """Segment the wavelet coefficients, encode each, then classify.

        Args:
            x_wvt (torch.Tensor): shape: (subjects, wavelets, regions)

        Returns:
            y, h, loss_recon
        """

        x = x_wvt[:, :self.n_wavelets, :]  # -> shape: (subjects, wavelets, regions)

        x_segments = x.unfold(1, self.segment_length, self.segment_length)
        x_segments = x_segments.reshape(-1, self.segment_length, self.n_regions)
        # -> shape: (subjects * segments, regions, segment_length)
        x_segments = x_segments.permute(0, 2, 1)

        h, _x_segments_recon, loss_recon = self.feature_extractor(x_segments)

        if self.phase == 'finetune':
            y = self.cls_head(h)
            y = y.reshape(x.size(0), -1, 2)
            y = y.mean(dim=1)
        else:
            y = None

        return y, h, loss_recon
