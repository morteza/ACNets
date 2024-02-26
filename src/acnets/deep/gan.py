import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ._fine_tunable import FineTunable


class Generator(pl.LightningModule):

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
            *self.block(latent_size, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            nn.Linear(512, input_size),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        is_valid = self.model(x)
        return is_valid


class GAN(FineTunable):
    def __init__(self, input_size, latent_size, lr=.0002, b1=.5, b2=.999):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_size, input_size)
        self.discriminator = Discriminator(input_size)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        x = batch[0]

        print(x.shape)

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_size)
        z = z.type_as(x)

        # generate
        self.toggle_optimizer(optimizer_g)
        self.generated_x = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log('g_loss', g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(x.size(0), 1)
        valid = valid.type_as(x)

        real_loss = F.binary_cross_entropy(self.discriminator(x), valid)

        # how well can it label as fake?
        fake = torch.zeros(x.size(0), 1)
        fake = fake.type_as(x)

        fake_loss = F.binary_cross_entropy(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr: float = self.hparams.lr
        b1: float = self.hparams.b1
        b2: float = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
