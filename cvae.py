import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, utils, mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tensorflow_probability as tfp


physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# CONTROL PARAMETERS
TRAIN = False
RESUME = False
SAVE_VAE = True
GENERATE = False
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/cvae/cvae'


# HYPER-PARAMETERS
EPOCHS = 201
LATENT_DIM = 512
# LEARNING_RATE = 0.0005
LEARNING_RATE = 0.0001
DECAY_RATE = 0.9
BATCH_SIZE = 128
DROPOUT = 0.05
# DROPOUT = 0.2
DIFFICULITY = 1.3
MODIFIER = 0
train_size = 50000
test_size = 10000

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        # epsilon = tf.cast(epsilon, tf.float16)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # return z_mean + tf.exp(tf.cast(0.5, tf.float16) * z_log_var)

def generate_and_save_images(cvae, epoch, test_sample, save_folder, save, generated_imgs_folder):

    if save:
        print("SAVING WEIGHTS AND IMAGES")
        cvae.save_weights(VAE_LOCATION, overwrite=True)

    n = 10
    # test_sample = tf.cast(test_sample, tf.float32)
    x = cvae.convbase1(test_sample[0:100])
    x = cvae.convbase2(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    z = sampler(z_mean, z_log_var)
    decoded_imgs = cvae.decoder.predict(z)
    # decoded_imgs = tf.cast(decoded_imgs, tf.float32)

    
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_sample[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("decoded")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(generated_imgs_folder+'image_at_epoch_{:04d}.png'.format(epoch))
    plt.clf()

class generate_save_callback(keras.callbacks.Callback):
    def __init__(self, test_samples, **kwargs):
        super().__init__(**kwargs)
        self.test_samples = test_samples
   
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            generate_and_save_images(self.model, epoch, self.test_samples, VAE_LOCATION, SAVE_VAE)

def decode(cvae, x):
    x = cvae.convbase1.predict(x)
    x = cvae.convbase2.predict(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    z = sampler(z_mean, z_log_var)

    return cvae.decoder.predict(z)

def noisy_decode(cvae, x):
    x = cvae.convbase1.predict(x)
    x = cvae.convbase2.predict(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    norm = tfp.distributions.Normal(0, 1)
    y_mean = norm.sample((50000, LATENT_DIM))
    y_log_var = norm.sample((50000, LATENT_DIM))
    z = sampler(z_mean+0.1*y_mean, z_log_var+0.1*y_log_var)

    return cvae.decoder.predict(z)

def encode(cvae, x):
    x = cvae.convbase1.predict(x)
    x = cvae.convbase2.predict(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    z = sampler(z_mean, z_log_var)

    return z

def plot_latent_images(cvae, x, latent_dim, n=10, digit_size=32):
    """Plots n x n digit images decoded from the latent space."""
    # plt.clf()
    # plt.show(block = False)
    
    norm = tfp.distributions.Normal(0, 1)

    x = x[0:n]
    o_x = x
    image_width = digit_size*n
    image_height = image_width

    # z_mean = norm.sample((n, latent_dim))
    # z_log_var = norm.sample((n, latent_dim))
    # z = cvae.sampler(z_mean, z_log_var)
    # z_decoded = cvae.decoder.predict(z)

    z = norm.sample((n, latent_dim))
    z_decoded = cvae.decoder.predict(z)

    # x_decoded = cvae.decode(x)
    x = cvae.convbase1.predict(x)
    x = cvae.convbase2.predict(x)
    z_mean, z_log_var = cvae.encoder.predict(x)
    sampler = Sampler()
    s = sampler(z_mean, z_log_var)
    x_decoded = cvae.decoder.predict(s)

    noisy_s = sampler(z_mean+0.8*z, z_log_var)
    noisy_x_decoded = cvae.decoder.predict(noisy_s)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
    # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(o_x[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_decoded[i])
        # plt.imshow(z_decoded[i])
        # plt.imshow(noisy_x_decoded[i])

        plt.title("z reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    
    plt.show()
    # plt.flush_events()

# CVAE CLASS

class CVAE(keras.Model):
    def __init__(self, convbase1, convbase2, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        # super().__init__(self, convbase1, convbase2, encoder, decoder,**kwargs)
        self.convbase1 = convbase1
        self.convbase2 = convbase2
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.test_total_loss_tracker = keras.metrics.Mean(name="test_total_loss")
        self.test_reconstruction_loss_tracker = keras.metrics.Mean(
            name="test_reconstruction_loss")
        self.test_kl_loss_tracker = keras.metrics.Mean(name="test_kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x = data
            # x = self.convbase_input(data)
            # x = data_augmentation(data)
            # x = layers.Rescaling(1./255)(x)
            x = self.convbase1(x)
            x = self.convbase2(x)
            z_mean, z_log_var = self.encoder(x)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
                # tf.reduce_sum(
                #     tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=data),
                #     axis=(1, 2)
                # )
            )

            kl_loss = 0.5*(tf.square(z_mean) + tf.exp(z_log_var) - 2*z_log_var - 1)
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    @tf.function
    def test_step(self, data):
        x = data

        x = self.convbase1(x)
        x = self.convbase2(x)
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)
            )
            # tf.reduce_sum(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=data),
            #     axis=(1, 2)
            # )
        )

        test_kl_loss = 0.5*(tf.square(z_mean) + tf.exp(z_log_var) - 2*z_log_var - 1)
        test_total_loss = reconstruction_loss + tf.reduce_mean(test_kl_loss)

        self.test_total_loss_tracker.update_state(test_total_loss)
        self.test_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.test_kl_loss_tracker.update_state(test_kl_loss)

        return {
            "test_total_loss": self.test_total_loss_tracker.result(),
            "test_reconstruction_loss": self.test_reconstruction_loss_tracker.result(),
            "test_kl_loss": self.test_kl_loss_tracker.result(),
        }
    @tf.function
    def build_cvae(self, x):
        input = x
        x = self.convbase1(x)
        x = self.convbase2(x)
        z_mean, z_log_var = self.encoder(x)
        sampler = Sampler()
        z = sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        cvae = keras.Model(input, reconstruction, name="cvae")
        return cvae
    @tf.function
    def decode(self, x):
        x = self.convbase1.predict(x)
        x = self.convbase2.predict(x)
        z_mean, z_log_var = self.encoder.predict(x)
        sampler = Sampler()
        z = sampler(z_mean, z_log_var)

        return self.decoder.predict(z)
    @tf.function
    def noisy_decode(self, x):
        x = self.convbase1.predict(x)
        x = self.convbase2.predict(x)
        z_mean, z_log_var = self.encoder.predict(x)
        sampler = Sampler()
        norm = tfp.distributions.Normal(0, 1)
        y_mean = norm.sample((50000, LATENT_DIM))
        y_log_var = norm.sample((50000, LATENT_DIM))
        z = sampler(z_mean+0.1*y_mean, z_log_var+0.1*y_log_var)

        return self.decoder.predict(z)
    @tf.function
    def encode(self, x):
        x = self.convbase1.predict(x)
        x = self.convbase2.predict(x)
        z_mean, z_log_var = self.encoder.predict(x)
        sampler = Sampler()
        z = sampler(z_mean, z_log_var)

        return z
