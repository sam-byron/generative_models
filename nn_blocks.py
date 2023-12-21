import math
from tensorflow.keras import layers, initializers

# NOTE: In order to minimize checkerboard artificating, we must make sure that filter_size is divisible by stride_size

def add_residual_block(x, filters, num_layers=1, strides=1, activation="relu", dropout=0, difficulity=1, filter_size=4):
    residual = x
    initializer = initializers.GlorotNormal()

    filters = math.floor(filters*difficulity)
    for l in range(num_layers):
        x = layers.Conv2D(filters, filter_size, kernel_initializer = initializer, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(activation)(x)
        # if l+1 < num_layers-1:
        #     if filters != residual.shape[-1]:
        #         residual = layers.Conv2D(filters, 1)(residual)
        #     x = layers.add([x, residual])
            # x = layers.Concatenate(axis=3)([x, residual])
        # residual = x
    x = layers.SpatialDropout2D(dropout)(x)
    if filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    if strides > 1:
        x = layers.Conv2D(filters, 3, strides=strides, kernel_initializer = initializer, padding="same")(x)

    return x

def transpose_res_block(x, filters, num_layers=1, strides=1, activation="relu", dropout=0, difficulity=1, filter_size=4):
    residual = x
    initializer = initializers.GlorotNormal()
    filters = math.floor(filters*difficulity)


    for l in range(num_layers):
        if l == 0:
            x = layers.Conv2DTranspose(filters, filter_size, strides=strides, kernel_initializer = initializer, padding="same")(x)
        else:
            x = layers.Conv2D(filters, 3, kernel_initializer = initializer, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(activation)(x)
        # if filters != residual.shape[-1]:
        #     residual = layers.Conv2D(filters, 1)(residual)
        # if l==0 and strides > 1:
        #     residual = layers.Conv2DTranspose(filters, 1, strides=strides)(residual)
        # x = layers.add([x,  residual])
        # x = layers.Concatenate(axis=3)([x, residual])
        # residual = x
    if filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.SpatialDropout2D(dropout)(x)
    if strides > 1:
        residual = layers.Conv2DTranspose(filters, 1, strides=strides)(residual)
    x = layers.add([x,  residual])
    
    return x