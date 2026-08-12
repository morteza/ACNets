"""Generative models, re-exported from `acnets.deep`.

`gan.py` existed in both packages because the two branches wrote it against
different frameworks -- PyTorch Lightning here, Keras 3 there -- not because
they modelled anything different. Both are now one plain-PyTorch
implementation in `acnets.deep.gan`, and `conditional_gan.py` moved alongside
it.

`hierarchical_temporal.py` stays here: it is the learnable-parcellation stub
and belongs to this line of work.
"""

from acnets.deep.conditional_gan import ConditionalGAN
from acnets.deep.gan import GAN

from .hierarchical_temporal import HierarchicalTemporalModel

__all__ = ['GAN', 'ConditionalGAN', 'HierarchicalTemporalModel']
