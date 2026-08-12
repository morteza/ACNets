"""Conditional GAN: the generator is told which group to synthesise.

The difference from `gan.GAN` is where the label enters. There the label only
shapes the discriminator's target; here it is concatenated to the latent vector,
so the generator produces a sample *of a requested class* rather than a sample
of the data in general. That is what makes it usable for augmenting a small
group, which is the reason it was written.

Ported from `src/multimodal/models/conditional_gan.py`, which was Keras 3 on the
PyTorch backend and reached into `keras.Model.trainable_weights` to apply
gradients by hand. Two bugs did not survive the port, both noted inline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, input_dim=15, latent_dim=64, n_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Generate one sample per requested label."""
        noise = torch.randn(y.shape[0], self.latent_dim,
                            device=y.device, dtype=torch.get_default_dtype())
        conditioned = torch.cat(
            [noise, F.one_hot(y, self.n_classes).to(noise.dtype)], dim=1)
        return self.model(conditioned)


class Discriminator(nn.Module):
    def __init__(self, input_dim=15, n_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # 0 = generated, 1..C = real. Logits, not softmax: cross_entropy
            # normalises internally, and the Keras version's softmax layer meant
            # the loss was applied to an already-normalised distribution.
            nn.Linear(128, n_classes + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConditionalGAN(nn.Module):
    def __init__(self, input_dim=15, latent_dim=64, n_classes=2,
                 lr=2e-4, b1=0.5, b2=0.999):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.lr, self.b1, self.b2 = lr, b1, b2

        self.generator = Generator(input_dim, latent_dim, n_classes)
        self.discriminator = Discriminator(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Synthesise data for the given labels."""
        return self.generator(y)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d]

    def training_step(self, batch, optimizers, device) -> dict:
        optimizer_g, optimizer_d = optimizers

        x_real, y_real = batch[0].to(device), batch[1].to(device)
        batch_size = y_real.shape[0]

        # discriminator: real batch keeps its class (shifted by one), generated
        # batch is class 0.
        optimizer_d.zero_grad()
        x_fake = self.generator(y_real).detach()
        x = torch.cat([x_real, x_fake], dim=0)
        # The Keras version wrote `zeros(batch_size) * n_classes`, which is
        # zeros times anything, so it happened to be right by accident. Written
        # as what it means.
        y_fake = torch.zeros(batch_size, dtype=torch.long, device=device)
        y = torch.cat([y_real + 1, y_fake], dim=0)

        d_loss = F.cross_entropy(self.discriminator(x), y)
        d_loss.backward()
        optimizer_d.step()

        # generator: make the discriminator name the requested class.
        optimizer_g.zero_grad()
        y_pred = self.discriminator(self.generator(y_real))
        # The Keras version one-hot encoded y_real over n_classes + 1 without
        # the +1 shift, so the generator was rewarded for producing samples the
        # discriminator labelled as *the class below* the requested one -- and
        # class 0 means generated, so class 1's target was "look fake".
        g_loss = F.cross_entropy(y_pred, y_real + 1)
        g_loss.backward()
        optimizer_g.step()

        return {
            'd_loss': float(d_loss.detach()),
            'g_loss': float(g_loss.detach()),
            'combined_loss': float((d_loss + g_loss).detach()),
        }

    @torch.no_grad()
    def validation_step(self, batch, device) -> dict:
        x, y = batch[0].to(device), batch[1].to(device)
        predicted = self.discriminator(x).argmax(dim=1)
        return {'accuracy': float((predicted == y + 1).float().mean())}
