"""Semi-supervised GAN over connectivity features.

The discriminator has `n_classes + 1` outputs: class 0 is "generated", classes
1..C are the real groups. That single design covers both uses this repo had for
a GAN, which is why there is now one implementation rather than two. Given
labels it is a semi-supervised classifier that happens to train against a
generator; given none it degenerates to an ordinary real/fake GAN, with the
real target spread uniformly over the C real classes.

This replaces `src/multimodal/models/gan.py`, which was the same architecture
written in Keras and reached the same place by a different route. The two
existed because they were written on different branches, not because they did
different things.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._fine_tunable import FineTunable


class Generator(nn.Module):

    def block(self, input_size: int, output_size: int, normalize=True):
        layers = []
        layers.append(nn.Linear(input_size, output_size))

        if normalize:
            layers.append(nn.BatchNorm1d(output_size, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def __init__(self, latent_size, input_size):
        super().__init__()

        self.model = nn.Sequential(
            *self.block(latent_size, 64, normalize=False),
            *self.block(64, 128),
            *self.block(128, 256),
            nn.Linear(256, input_size),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, n_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, n_classes + 1),  # 1 fake (0) + n real [1-n)
        )

    def forward(self, x):
        y = self.model(x)
        return y


class GAN(FineTunable):
    """Semi-supervised GAN.

    Note on the discriminator's final layer: it returns logits, where the
    Lightning version ended in a Sigmoid and then fed the result to
    `F.cross_entropy`. That was wrong -- cross_entropy applies its own
    log_softmax, so the sigmoid was squashing the range before a second
    normalisation and flattening the gradients. Removing it is a behaviour
    change, and an intended one.
    """

    def __init__(self, input_size, latent_size, n_classes, lr=.0002, b1=.5, b2=.999):
        super().__init__()
        self.save_hyperparameters(input_size=input_size, latent_size=latent_size,
                                  n_classes=n_classes, lr=lr, b1=b1, b2=b2)

        self.input_size = input_size
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.lr, self.b1, self.b2 = lr, b1, b2

        self.generator = Generator(latent_size, input_size)
        self.discriminator = Discriminator(input_size, n_classes=n_classes)

    def forward(self, z):
        x = self.generator(z)
        return x

    def _real_targets(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """Target distribution for a batch the discriminator should call real.

        With labels this is one-hot on the true class (shifted by one, since 0
        means generated). Without them the mass is spread evenly over the real
        classes, which says "some real class, no opinion which".
        """
        if y is not None:
            targets = torch.zeros(x.size(0), self.n_classes + 1, device=x.device)
            return targets.scatter(1, y.unsqueeze(1), 1)

        row = torch.tensor([0.0] + [1.0 / self.n_classes] * self.n_classes,
                           device=x.device, dtype=x.dtype)
        return row.reshape(1, -1).repeat(x.size(0), 1)

    def training_step(self, batch, optimizers, device) -> dict:
        optimizer_g, optimizer_d = optimizers

        x = batch[0].to(device)
        y = batch[1].to(device) + 1 if len(batch) > 1 else None  # 0 is reserved for fake
        x = x.view(x.size(0), -1)

        if x.shape[0] == 1:
            # BatchNorm1d in the generator cannot compute batch statistics from
            # a single sample.
            return {}

        z = torch.randn(x.shape[0], self.latent_size, device=device, dtype=x.dtype)

        # generator: push the discriminator towards calling generated data real
        optimizer_g.zero_grad()
        real_targets = self._real_targets(x, y)
        real_pred = self.discriminator(self(z))
        g_loss = F.cross_entropy(real_pred, real_targets)
        g_loss.backward()
        optimizer_g.step()

        # discriminator: real data as its class, generated data as class 0
        optimizer_d.zero_grad()
        real_targets = self._real_targets(x, y)
        real_loss = F.cross_entropy(self.discriminator(x), real_targets)

        fake_targets = torch.zeros(x.size(0), self.n_classes + 1,
                                   device=device, dtype=x.dtype)
        fake_targets[:, 0] = 1.0
        fake_pred = self.discriminator(self(z).detach())
        fake_loss = F.cross_entropy(fake_pred, fake_targets)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        return {
            'g_loss': float(g_loss.detach()),
            'd_loss': float(d_loss.detach()),
            'accuracy_real': float(
                (real_pred.argmax(1) == real_targets.argmax(1)).float().mean()),
            'accuracy_fake': float((fake_pred.argmax(1) == 0).float().mean()),
        }

    @torch.no_grad()
    def validation_step(self, batch, device) -> dict:
        x = batch[0].to(device)
        x = x.view(x.size(0), -1)

        y_pred = self.discriminator(x)

        if len(batch) == 1:
            correct = (y_pred.argmax(1) != 0).float().mean()
        else:
            y = batch[1].to(device) + 1
            correct = (y_pred.argmax(1) == y).float().mean()

        return {'accuracy': float(correct)}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d]
