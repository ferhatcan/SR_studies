import os
import tensorflow as tf
import numpy as np
from tensorflow.python.data.experimental import AUTOTUNE


# Author: Ferhat Can ATAMAN
# Date: 17/02/2020
# It is written to use thermal dataset that is saved as png format
# train&test dataset should be created explicitly.
# @todo cache usage have not been implemented yet.
# @todo make some tests to sure there is no error
# @todo when the dataset is used for ircnn-denoise model, gt-input model should be handled
class IR_DATASET:
    def __init__(self,
                 images_dir,
                 scale=1,
                 validation_size=100,
                 include_noise=True,
                 noise_sigma=10,
                 noise_mean=0,
                 hr_shape=[96, 96],
                 hr_model="image",
                 downgrade="bicubic",
                 subset="noise_model",
                 caches_dir="./caches"):

        self.scale = scale
        self.subset = subset
        self.caches_dir = caches_dir
        self.images_dir = images_dir
        self.downgrade = downgrade
        self.validation_size = validation_size
        self.include_noise = include_noise
        self.noise_sigma = noise_sigma
        self.noise_mean = noise_mean
        self.hr_shape = hr_shape
        self.hr_model = hr_model

        if self.subset == "noise_model":
            self.scale = 1
            self.include_noise = True

        os.makedirs(caches_dir, exist_ok=True)

    def dataset(self, batch_size=8, repeat_count=None, random_transform=True):
        path_ds = tf.data.Dataset.list_files(self.images_dir)
        ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        if random_transform:
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def preprocess_image(self, image):
        hr_image = tf.image.decode_jpeg(image, channels=1)
        hr_image = tf.image.random_crop(hr_image, [self.hr_shape[0], self.hr_shape[1], 1])
        lr_image = tf.image.resize(hr_image, [self.hr_shape[0] // self.scale, self.hr_shape[1] // self.scale], method=tf.image.ResizeMethod.BICUBIC)
        if self.include_noise:
            noise = np.random.normal(self.noise_mean, self.noise_sigma, lr_image.shape)
            lr_image_noised = lr_image + noise
            lr_image_noised = tf.clip_by_value(lr_image_noised, 0, 255)
            lr_image_noised = tf.round(lr_image_noised)
            if self.subset == "noise_model":
                hr_image = noise
        else:
            lr_image_noised = lr_image
            hr_image = hr_image

        if self.hr_model == "difference":
            lr_image_up = tf.image.resize(lr_image_noised, [self.hr_shape[0], self.hr_shape[1]],
                                          method=tf.image.ResizeMethod.BICUBIC)
            lr_image_up = tf.clip_by_value(lr_image_up, 0, 255)
            lr_image_up = tf.cast(lr_image_up, tf.float32)
            hr_image = tf.cast(hr_image, tf.float32)
            hr_image = hr_image - lr_image_up
            hr_image = hr_image  # / 255

        # lr_image /= 255.0
        # hr_image /= 255.0
        # print(lr_image.shape, hr_image.shape)
        return lr_image_noised, hr_image


# --------------------------------------------------------------
#       Transformations
# --------------------------------------------------------------
def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))