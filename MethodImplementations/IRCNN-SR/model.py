import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, ReLU, BatchNormalization, Conv2DTranspose
from tensorflow.python.keras.models import Model

OUTPUT_CHANNELS = 1


def ircnn_denoise(num_filters=64, noise_level=12):
    x_in = Input(shape=(None, None, 1))
    x = Lambda(lambda t: t / 127.5 - 1)(x_in)

    dilation_array = [1, 2, 3, 4, 3, 2, 1]

    x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_array[0])(x)
    x = ReLU()(x)

    for i in range(1, len(dilation_array) - 1):
        x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_array[i])(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    x = Conv2D(1, 3, padding="same", dilation_rate=dilation_array[-1])(x)

    # x = Lambda(lambda t: (t + 1) * 127.5)(x)
    return Model(x_in, x, name=f"ircnn_denoise_nl{noise_level}")


def residual_sr(scale=2):
    inputs = Input(shape=(None, None, 1))  # Input shape should be larger than 128,128,1
    inputs_norm = Lambda(lambda t: t / 127.5 - 1)(inputs)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 64, 64, 64)
        downsample(128, 4),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 128, 128, 3)

    x = inputs_norm

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    x = x + inputs_norm

    if scale == 2:
        x = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    elif scale == 4:
        x = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
        x = Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer)(x)

    x = Lambda(lambda t: (t + 1) * 127.5)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def residual_sr_v2(scale=2):
    inputs = Input(shape=(64, 64, 1))  # Input shape should be larger than 128,128,1
    inputs_norm = Lambda(lambda t: t / 127.5 - 1)(inputs)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 64, 64, 64)
        downsample(128, 4),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 128, 128, 3)

    x = inputs_norm

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = upsample(64, 4)(x)
    if scale == 2:
        x = upsample(32, 4)(x)
    elif scale == 4:
        x = upsample(32, 4)(x)
        x = upsample(16, 4)(x)
    x = last(x)
    x = Lambda(lambda t: (t + 1) * 127.5)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def residual_sr_v3(scale=2):
    inputs = Input(shape=(64, 64, 1))  # Input shape should be larger than 128,128,1
    inputs_norm = Lambda(lambda t: t / 127.5 - 1)(inputs)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 64, 64, 64)
        downsample(128, 4),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer)  # (bs, 128, 128, 3)

    x = inputs_norm

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = upsample(64, 4)(x)
    if scale == 2:
        x = Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    elif scale == 4:
        x = Conv2DTranspose(32, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
        x = Conv2DTranspose(16, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result
