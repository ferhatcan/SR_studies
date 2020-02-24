import numpy as np
import os
from dataset import IR_DATASET
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from model import ircnn_denoise
from train import ircnn_denoise_trainer
from train import show_HR_SR_pair
import tensorflow as tf

from model import residual_sr_v3
from train import residual_sr_trainer


SHOW_SAMPLE = False
SIZE_OF_VALIDATION = 100

scale = 4
total_iter = 10
in_iter = 5
alpha = 0.01
noise_mean = 0
model_sigmas = np.logspace(np.log10(12*scale), np.log10(scale), total_iter)
noise_sigmas = model_sigmas // 2
noise_sigmas = [min(25, max(x, 1)) for x in noise_sigmas]

data_path_IR = "/home/ferhatcan/Desktop/challengedataset/train/**/*.jpg"
data_path_test_IR = "/home/ferhatcan/Desktop/challengedataset/test/640_flir_hr/*.jpg"

# -----------------------
#       Dataset
# -----------------------
train_ds = IR_DATASET(images_dir=data_path_IR, scale=scale, subset="residual_sr",
                      noise_mean=noise_mean, noise_sigma=noise_sigmas[-1], hr_model="difference")

train_ds = train_ds.dataset(batch_size=1)

validation_ds = train_ds.take(SIZE_OF_VALIDATION)
train_ds = train_ds.skip(SIZE_OF_VALIDATION)

test_ds = IR_DATASET(images_dir=data_path_test_IR, scale=scale, subset="residual_sr",
                     noise_mean=noise_mean, noise_sigma=noise_sigmas[3], hr_shape=[64*scale, 64*scale], hr_model="difference")

test_ds = test_ds.dataset(batch_size=1, random_transform=False)

# -----------------------
#       Trainer
# -----------------------
trainer_sr = residual_sr_trainer(model=residual_sr_v3(scale=scale),
                                 checkpoint_dir=f'.ckpt/residual_sr_{scale}_v3_difference_01',
                                 learning_rate=PiecewiseConstantDecay(boundaries=[150000], values=[2e-4, 1e-6]))

for lr, hr in test_ds.take(10):
    sr = trainer_sr.SR_return(lr)
    output = tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
    hr = hr + output
    output = output + sr
    for i in range(total_iter):
        # Location of model weights (needed for demo)
        weights_dir = f'weights/ircnn_denoise_{noise_sigmas[i] // 1}'
        weights_file = os.path.join(weights_dir, 'weights.h5')
        os.makedirs(weights_dir, exist_ok=True)

        # -----------------------
        #       Trainer
        # -----------------------
        trainer = ircnn_denoise_trainer(model=ircnn_denoise(num_filters=24, noise_level=noise_sigmas[i]),
                                        checkpoint_dir=f'.ckpt/ircnn_denoise_{noise_sigmas[i] // 1}',
                                        learning_rate=PiecewiseConstantDecay(boundaries=[15000], values=[1e-4, 5e-5]))

        # Restore from checkpoint with highest PSNR
        trainer.restore()

        # @todo there is an error in this part (there makes more noisy output)
        for j in range(in_iter):
            downscaled_ref = tf.image.resize(output, [hr.shape[1] // scale, hr.shape[2] // scale],
                                             method=tf.image.ResizeMethod.BICUBIC)
            sr = trainer_sr.SR_return(downscaled_ref)
            output = output + alpha * sr
            # show_HR_SR_pair(lr, hr, output)
            """
            downscaled_ref = tf.image.resize(output, [hr.shape[1] // scale, hr.shape[2] // scale], method=tf.image.ResizeMethod.BICUBIC)
            error = tf.image.resize(lr - downscaled_ref, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
            output = output + alpha * error
            # show_HR_SR_pair(lr - downscaled_ref, downscaled_ref)
            """

        # output = tf.clip_by_value(output, 0, 255)
        # output = tf.round(output)
        noise = trainer.SR_return(output)
        output = output - noise
        # output = tf.clip_by_value(output, 0, 255)
        # output = tf.round(output)
        # show_HR_SR_pair(noise, output)

    show_HR_SR_pair(lr, hr, output)