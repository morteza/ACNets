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
            *self.block(latent_size, 64, normalize=False),
            *self.block(64, 128),
            *self.block(128, 256),
            nn.Linear(256, input_size),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(pl.LightningModule):
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y


class GAN(FineTunable):
    def __init__(self, input_size, latent_size, n_classes, lr=.0002, b1=.5, b2=.999):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_size, input_size)
        self.discriminator = Discriminator(input_size, n_classes=n_classes)

    def forward(self, z):
        x = self.generator(z)
        return x

    def training_step(self, batch):

        x = batch[0]
        y = batch[1] + 1 if len(batch) > 1 else None   # plus 1 because fake=0
        x = x.view(x.size(0), -1)
        if x.shape[0] == 1:
            print('Batch size is 1, skipping...')
            return

        optimizer_g, optimizer_d = self.optimizers()

        # generate
        z = torch.randn(x.shape[0], self.hparams.latent_size).type_as(x)
        self.toggle_optimizer(optimizer_g)

        if y is not None:
            real = torch.zeros(x.size(0), self.hparams.n_classes + 1).type_as(x)
            real = real.scatter(1, y.unsqueeze(1), 1)
        else:
            real = torch.Tensor([0] + [1/self.hparams.n_classes] * self.hparams.n_classes)
            real = real.type_as(x).reshape(1, -1).repeat(x.size(0), 1)

        real_pred = self.discriminator(self(z))
        g_loss = F.cross_entropy(real_pred, real)
        self.log('train/g_loss', g_loss, prog_bar=True)
        real_acc = (real_pred.argmax(1) == real.argmax(1)).float().mean()
        self.log('train/accuracy_real', real_acc)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        if y is not None:
            real = torch.zeros(x.size(0), self.hparams.n_classes + 1).type_as(x)
            real = real.scatter(1, y.unsqueeze(1), 1)
        else:
            real = torch.Tensor([0] + [1/self.hparams.n_classes] * self.hparams.n_classes)
            real = real.type_as(x).reshape(1, -1).repeat(x.size(0), 1)

        real_loss = F.cross_entropy(self.discriminator(x), real)

        # how well can it label as fake?
        # fake = torch.zeros(x.size(0), 1).type_as(x)
        fake = torch.Tensor([1] + [0] * self.hparams.n_classes).reshape(1, -1).repeat(x.size(0), 1)
        fake = fake.to(x.device)
        fake_pred = self.discriminator(self(z).detach())
        fake_loss = F.cross_entropy(fake_pred, fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log('train/d_loss', d_loss, prog_bar=True)
        fake_acc = (fake_pred.argmax(1) == 0).float().mean()
        self.log('train/accuracy_fake', fake_acc)

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x = x.view(x.size(0), -1)

        y_pred = self.discriminator(x)

        if len(batch) == 1:
            accuracy = (y_pred.argmax(1) != 0).float().mean()
        else:
            y = batch[1] + 1
            accuracy = (y_pred.argmax(1) == y).float().mean()

        self.log('val/accuracy', accuracy)

    def configure_optimizers(self):
        lr: float = self.hparams.lr
        b1: float = self.hparams.b1
        b2: float = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
