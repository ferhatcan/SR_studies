import time
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/ircnn'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            # print(lr.numpy().shape, hr.numpy().shape)
            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(f'{step}/{steps}: loss = {loss_value.numpy():.6f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = tf.reduce_mean(tf.abs(hr - sr))  # self.loss(hr * 255, sr * 255)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        psnr_values = []
        for lr, hr in dataset:
            sr = resolve(self.checkpoint.model, lr)
            # show_HR_SR_pair(hr, sr)
            # @todo remove if use method different than differential method
            # sr = sr * 255 + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
            # sr = sr + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
            # hr = hr * 255 + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
            # hr = hr + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
            psnr_value = psnr_np(sr.numpy().squeeze(), hr.numpy().squeeze())  # psnr(hr, sr)[0]
            psnr_values.append(psnr_value)
        return tf.reduce_mean(psnr_values)

    def SR_return(self, lr):
        return resolve(self.checkpoint.model, lr)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    # sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    # sr_batch = tf.round(sr_batch)
    # sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def cast(img):
    img = np.clip(img, 0, 255)
    img = np.rint(img)
    img = img.astype(np.uint8)
    return img


def show_HR_SR_pair(lr, hr, sr):
    if hr.shape[0] != 1:
        lr = lr[1, :, :, :]
        hr = hr[1, :, :, :]
        sr = sr[1, :, :, :]
    lr = tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
    fig = plt.figure()
    plt.imshow(hr.numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    fig2 = plt.figure()
    plt.imshow(sr.numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    fig3 = plt.figure()
    plt.imshow(lr.numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    plt.close('all')
    print(np.mean(np.abs(sr.numpy().squeeze() - hr.numpy().squeeze())))
    print(psnr_np(sr.numpy().squeeze(), hr.numpy().squeeze()))
    print(psnr_np(lr.numpy().squeeze(), hr.numpy().squeeze()))


def show_noise_pair(hr, sr, lr):
    hr = hr.numpy().squeeze()
    lr = lr.numpy().squeeze()
    sr = sr.numpy().squeeze()
    fig = plt.figure()
    plt.imshow(lr, cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    fig = plt.figure()
    plt.imshow(cast(lr-hr), cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    fig2 = plt.figure()
    plt.imshow(cast(lr-sr), cmap='gray', vmin=0, vmax=255)
    plt.show()
    time.sleep(1)
    plt.close('all')
    print(np.mean(np.abs(sr - hr)))
    print(psnr_np(lr-sr, lr-hr))


def psnr_np(img1, img2):
    mse = np.mean((img1 - img2) * (img1 - img2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class ircnn_denoise_trainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class residual_sr_trainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)