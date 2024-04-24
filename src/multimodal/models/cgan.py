"""A simple CGAN implementation using Keras and PyTorch.
"""

import keras
import torch
from keras import layers, ops

class CGAN(keras.Model):
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
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = keras.Sequential([
            # NOTE +n for one-hot encoded class
            keras.Input(shape=(self.latent_dim + self.n_classes + 1,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.input_dim)
        ], name='generator')

        self.discriminator = keras.Sequential([
            keras.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            # NOTE +1 for fake/real
            layers.Dense(self.n_classes + 1, activation='sigmoid')
        ], name='discriminator')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def compile(self, loss, d_optimizer, g_optimizer):
        super().compile(loss=loss)
        self.loss_fn = loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        # self.additional_metrics = metrics

    def call(self, x, training):
        return self.discriminator(x).argmax(axis=1)
        

    def train_step(self, real_data):
        (x_real, y_real) = real_data
        y_real = ops.one_hot(y_real + 1, self.n_classes + 1)

        batch_size = ops.shape(y_real)[0]

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    seed=self.seed_generator)

        noise = ops.concatenate([
            noise,
            y_real
        ], axis=1)

        x_fake = self.generator(noise)

        x = ops.concatenate([x_real, x_fake], axis=0)

        y_fake = ops.one_hot(ops.zeros((batch_size,)), self.n_classes + 1)
        y = ops.concatenate([y_fake, y_real], axis=0)
        # NOTE y=[1,0,0] for fake, [0,1,0] for C=1 y=[0,0,1] for C=2

        # 0=fake, 1..C=real
        # trick that adds noise to labels
        y = y + 0.05 * keras.random.uniform(ops.shape(y))

        # train discriminator
        self.zero_grad()
        y_pred = self.discriminator(x, training=True)
        d_loss = self.loss_fn(y, y_pred)  # TODO use compute_loss?
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    seed=self.seed_generator)
        noise = ops.concatenate([
            noise,
            y_real
        ], axis=1)

        self.zero_grad()
        y_pred = self.discriminator(self.generator(noise))
        g_loss = self.loss_fn(y_real, y_pred)
        g_loss.backward()  # FIXME move it below?
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            'd_loss': self.d_loss_tracker.result(),
            'g_loss': self.g_loss_tracker.result(),
        }
