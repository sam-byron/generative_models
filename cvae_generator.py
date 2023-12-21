import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, metrics, utils, mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tensorflow_probability as tfp
from keras_cv.layers import BaseImageAugmentationLayer
from cvae import CVAE, generate_save_callback,plot_latent_images, generate_and_save_images, decode, encode, noisy_decode
from nn_blocks import add_residual_block, transpose_res_block


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# CONTROL PARAMETERS
TRAIN = False
RESUME = False
SAVE_VAE = False
GENERATE = True
VAE_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/cvae_generator/cvae_generator'
RECONSTRUCTED_IMGS_LOCATION = '/home/sambyron/engineering/ML/tensorflow/cifar10/cvae_generator/reconstructed_imgs/'


# HYPER-PARAMETERS
EPOCHS = 401
LATENT_DIM = 512
# LEARNING_RATE = 0.0005
LEARNING_RATE = 0.0001
DECAY_RATE = 1
BATCH_SIZE = 64
DROPOUT = 0.1
# DROPOUT = 0.2
DIFFICULITY = 1
train_size = 50000
test_size = 10000

        
# TRAINING USING VAE

# LOAD DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_dataset = (tf.data.Dataset.from_tensor_slices(x_train)
                 .shuffle(train_size).batch(BATCH_SIZE))
test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)
                .shuffle(test_size).batch(BATCH_SIZE))

# x_train = x_train[0:1000]

# x_train = x_train[0:10000]

DATA_SHAPE = x_train.shape[1:]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# ENCODER

# data_augmentation = keras.Sequential(
#         [
#             layers.RandomFlip("horizontal"),
#             layers.RandomBrightness(0.4),
#             layers.RandomRotation(0.1),
#             layers.RandomZoom((-0.1, 0.1)),
#             layers.RandomContrast(0.4),
#             # layers.RandomCrop(28,28),
#             layers.RandomTranslation(0.1,0.1)
#         ]
# )

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"), #GOOD learning rate
            layers.RandomBrightness(0.2), #GOOD learning rate
            layers.RandomRotation(0), #SLOW learning rate
            layers.RandomZoom((-0.3, 0.3)), #GOOD learning rate
            layers.RandomContrast(0.41), #GOOD learning rate
            layers.RandomCrop(32,32), #SLOW-AVG learning rate (poor acc performance?), smaller number of params
            layers.RandomTranslation(0.35,0.35), #AVG learning rate
        ]
)

x_train = layers.Rescaling(1./255)(x_train)
# augmented_x_train = data_augmentation(x_train)
x_test = layers.Rescaling(1./255)(x_test)

initializer = initializers.GlorotNormal()

convbase_inputs1 = keras.Input(shape=(32, 32, 3))
x = convbase_inputs1
x = add_residual_block(x, 64, 2, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)
out = add_residual_block(x, 128, 3, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)
convbase1 = keras.Model(convbase_inputs1, out, name="convbase1")
convbase1.summary()

convbase_inputs2 = keras.Input(shape=(8, 8, math.floor(128*DIFFICULITY)))
x = convbase_inputs2
out = add_residual_block(x, 256, 4, strides=1, dropout=DROPOUT, difficulity=DIFFICULITY)
convbase2 = keras.Model(convbase_inputs2, out, name="convbase2")
convbase2.summary()


encoder_inputs = keras.Input(shape=(8, 8, math.floor(256*DIFFICULITY)))
x = encoder_inputs
x = layers.Flatten()(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_initializer=initializer)(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_initializer=initializer)(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()
    
# DECODER
    
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(8 * 8 * math.floor(256*DIFFICULITY), activation="relu", kernel_initializer=initializer)(latent_inputs)
x = layers.Reshape((8, 8, math.floor(256*DIFFICULITY)))(x)

x = transpose_res_block(x, 256, 4, strides=1, dropout=DROPOUT, difficulity=DIFFICULITY)

x = transpose_res_block(x, 128, 3, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)

x = transpose_res_block(x, 64, 2, strides=2, dropout=DROPOUT, difficulity=DIFFICULITY)

decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same", kernel_initializer=initializer)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()

# RUN VAE

import numpy as np
import time
import matplotlib.pyplot as plt


cvae = CVAE(convbase1, convbase2, encoder, decoder)
cvae.compile()

# RESUME TRAINING MODEL
if RESUME:
    print("LOADING WEIGHTS AND RESUMING TRAINING")
    cvae.load_weights(VAE_LOCATION)
    # plot_latent_images(cvae, x_train[0:100])


lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=10000,
        decay_rate=DECAY_RATE)
cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), run_eagerly=True)

if TRAIN:

    for epoch in range(0, EPOCHS):
        start_time = time.time()
        for batch_x_train in train_dataset:
           batch_x_train = data_augmentation(batch_x_train)
           batch_x_train = layers.Rescaling(1./255)(batch_x_train)

           cvae.train_step(batch_x_train)
        

        loss = tf.keras.metrics.Mean()
        for batch_x_test in test_dataset:
            batch_x_test = layers.Rescaling(1./255)(batch_x_test)
            cvae.test_step(batch_x_test)
            loss = cvae.test_total_loss_tracker.result()
        end_time = time.time()
        print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}'
                .format(epoch, loss, end_time - start_time))
        if epoch % 5 == 0:
            generate_and_save_images(cvae, epoch, batch_x_test, VAE_LOCATION, False, RECONSTRUCTED_IMGS_LOCATION)
            if GENERATE:
                plot_latent_images(cvae, x_train[0:100], LATENT_DIM)
            if SAVE_VAE:
                print("SAVING WEIGHTS")
                cvae.save_weights(VAE_LOCATION, overwrite=True)

elif GENERATE:
    print("LOADING WEIGHTS TO GENERATE NEW IMAGES")
    cvae.load_weights(VAE_LOCATION)
    for i in range(10):
        plot_latent_images(cvae, x_train[i*10:120], LATENT_DIM)

#================================================================================