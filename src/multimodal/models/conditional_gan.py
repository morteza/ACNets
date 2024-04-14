# Conditional GAN using Keras 3
import keras


class ConditionalGAN(keras.Model):
    def __init__(self):
        super().__init__()
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flat(inputs)
        x = self.dense1(x)
        return self.dense2(x)
