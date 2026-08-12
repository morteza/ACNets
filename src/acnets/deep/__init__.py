"""Deep models, in plain PyTorch.

Every model here is an `nn.Module`. Training is driven by
`acnets.deep.training.Trainer`, which models plug into by implementing
`compute_loss` (single loss) or `training_step` (adversarial, two optimizers).
`FineTunable` adds the two-phase pretrain/finetune protocol on top.
"""

# Must precede every torch import below: XGBoost segfaults if it loads its
# OpenMP runtime after torch loads a different one. See acnets._openmp.
from .._openmp import preload as _preload_xgboost

_preload_xgboost()

from .conditional_gan import ConditionalGAN  # noqa: E402
from .diffusion import GaussianNormative, VectorDiffusion  # noqa: E402
from .gan import GAN  # noqa: E402
from .masked_model import MaskedModel  # noqa: E402
from .multihead_model import MultiHeadModel  # noqa: E402
from .multiview import MultiViewClassifier  # noqa: E402
from .timepoint_ae import LatentConnectivityClassifier, TimepointAE  # noqa: E402
from .torch_classifier import TorchMLPClassifier  # noqa: E402
from .training import MetricLogger, Trainer, accuracy, resolve_device  # noqa: E402
from .wavelet_model import WaveletModel  # noqa: E402

__all__ = [
    'GAN',
    'ConditionalGAN',
    'GaussianNormative',
    'LatentConnectivityClassifier',
    'MaskedModel',
    'MetricLogger',
    'MultiHeadModel',
    'MultiViewClassifier',
    'TimepointAE',
    'TorchMLPClassifier',
    'Trainer',
    'VectorDiffusion',
    'WaveletModel',
    'accuracy',
    'resolve_device',
]
