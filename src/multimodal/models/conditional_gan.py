"""A simple CGAN implementation using Keras and PyTorch.
"""

import keras
import torch
import torch.nn.functional as F
from keras import layers, ops

class Generator(keras.Model):
    def __init__(self, input_dim=15, latent_dim=64, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.seed_generator = keras.random.SeedGenerator(42)

        self.model = keras.Sequential([
            # NOTE 0..C-1
            keras.Input(shape=(self.latent_dim + self.n_classes,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.input_dim)
        ], name='generator')

        self.built = True

    def call(self, y, training=True):

        batch_size = ops.shape(y)[0]

        noise = keras.random.normal((batch_size, self.latent_dim))
        noise = ops.concatenate([
            noise,
            F.one_hot(y, self.n_classes)
        ], axis=1)

        return self.model(noise)

class Discriminator(keras.Model):
    def __init__(self, input_dim=15, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.model = keras.Sequential([
            keras.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            # NOTE 0 = fake, 1..C = real
            layers.Dense(self.n_classes + 1, activation='softmax')
        ], name='discriminator')

        self.built = True

    def call(self, x):
        return self.model(x)

class ConditionalGAN(keras.Model):
    def __init__(self,
                 input_dim=15,
                 latent_dim=64,
                 n_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.combined_loss_tracker = keras.metrics.Mean(name='combined_loss')
        self.accuracy = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = Generator(input_dim, latent_dim, n_classes)
        self.discriminator = Discriminator(input_dim, n_classes)
        self.built = True

    @property
    def metrics(self):
        return [self.accuracy]

    def compile(self, loss, g_optimizer, d_optimizer):
        super().compile(loss=loss)
        self.loss_fn = loss
        self.d_optimizer=g_optimizer
        self.g_optimizer=d_optimizer

    def call(self, x, training=False):
        pred = self.discriminator(x).argmax(axis=1)
        if training:
            return pred  # 0=fake, 1..C=real
        else:
            return pred - 1  # 0..C-1=real

    def train_step(self, real_data):
        (x_real, y_real) = real_data
        batch_size = ops.shape(y_real)[0]

        x_fake = self.generator(y_real)

        x = ops.concatenate([x_real, x_fake], axis=0)

        y_fake = ops.zeros((batch_size,), dtype=torch.int64) * self.n_classes
        y = ops.concatenate([y_fake, y_real + 1], axis=0)  # 0=fake, 1..C=real

        # FIXME trick that adds noise to labels
        # y = y + 0.05 * keras.random.uniform(ops.shape(y))

        # train discriminator
        self.zero_grad()
        y_pred = self.discriminator(x)
        d_loss = self.loss_fn(F.one_hot(y, self.n_classes + 1), y_pred)  # TODO use compute_loss?
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        self.zero_grad()
        y_pred = self.discriminator(self.generator(y_real))
        g_loss = self.loss_fn(F.one_hot(y_real, self.n_classes + 1), y_pred)
        g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.combined_loss_tracker.update_state(d_loss + g_loss)
        return {
            'd_loss': self.d_loss_tracker.result(),
            'g_loss': self.g_loss_tracker.result(),
            'combined_loss': self.combined_loss_tracker.result(),
        }
