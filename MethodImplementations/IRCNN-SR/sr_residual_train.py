import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import numpy as np
import os
from dataset import IR_DATASET
from model import residual_sr_v2
from train import residual_sr_trainer
from train import show_HR_SR_pair

scale = 4
noise_mean = 0
noise_sigma = 10

tf.keras.utils.plot_model(residual_sr_v2(scale=scale), show_shapes=True, dpi=64, to_file=f'residual_sr_{scale}_v2_direct01.png')


SHOW_SAMPLE = True
TRAINING = True
SIZE_OF_VALIDATION = 100
BATCH_SIZE = 16

data_path_IR = "/home/ferhatcan/Desktop/challengedataset/train/640_flir_hr/*.jpg"
data_path_test_IR = "/home/ferhatcan/Desktop/challengedataset/test/640_flir_hr/*.jpg"

# Location of model weights (needed for demo)
weights_dir = f'weights/residual_sr_{scale}_v2_direct01'
weights_file = os.path.join(weights_dir, 'weights.h5')
os.makedirs(weights_dir, exist_ok=True)

# -----------------------
#       Dataset
# -----------------------
train_ds = IR_DATASET(images_dir=data_path_IR, scale=scale, subset="residual_sr",
                      noise_mean=noise_mean, noise_sigma=noise_sigma, hr_shape=[64*scale, 64*scale], hr_model=" ")

train_ds = train_ds.dataset(batch_size=BATCH_SIZE)

validation_ds = train_ds.take(SIZE_OF_VALIDATION)
train_ds = train_ds.skip(SIZE_OF_VALIDATION)

test_ds = IR_DATASET(images_dir=data_path_test_IR, scale=scale, subset="residual_sr",
                     noise_mean=noise_mean, noise_sigma=noise_sigma, hr_shape=[64*scale, 64*scale], hr_model=" ")

test_ds = test_ds.dataset(batch_size=1, random_transform=False)

# -----------------------
#       Trainer
# -----------------------
trainer = residual_sr_trainer(model=residual_sr_v2(scale=scale),
                              checkpoint_dir=f'.ckpt/residual_sr_{scale}_v2_direct01',
                              learning_rate=PiecewiseConstantDecay(boundaries=[150000], values=[1e-4, 1e-6]))
if TRAINING:
    trainer.train(train_ds,
                  validation_ds,
                  steps=300000,
                  evaluate_every=500,
                  save_best_only=True)

# Restore from checkpoint with highest PSNR
trainer.restore()

# ------------------------
#       Evaluation
# ------------------------
# Evaluate model on full validation set
if TRAINING:
    psnrv = trainer.evaluate(test_ds.take(1))
    print(f'PSNR = {psnrv.numpy():3f}')

if SHOW_SAMPLE:
    for lr, hr in test_ds.take(10):
        sr = trainer.SR_return(lr)
        print(np.mean(np.abs(sr.numpy().squeeze() - hr.numpy().squeeze())))
        # sr = sr * 255 + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
        # sr = sr + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
        # sr = sr / 2
        print(np.mean(np.abs(hr.numpy().squeeze())))
        # hr = hr * 255 + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
        # hr = hr + tf.image.resize(lr, [hr.shape[1], hr.shape[2]], method=tf.image.ResizeMethod.BICUBIC)
        show_HR_SR_pair(lr, hr, sr)


# ------------------------
#       Save Weights
# ------------------------
# Save weights to separate location (needed for demo)
trainer.model.save_weights(weights_file)