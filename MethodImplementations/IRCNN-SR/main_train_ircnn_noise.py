import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import numpy as np
import os
from dataset import IR_DATASET
from model import ircnn_denoise
from train import ircnn_denoise_trainer
from train import show_noise_pair

scale = 1
total_iter = 30
noise_mean = 0
model_sigmas = np.logspace(np.log10(12*scale), np.log10(scale), total_iter)
noise_sigmas = model_sigmas // 2
noise_sigmas = [min(25, max(x, 1)) for x in noise_sigmas]

SHOW_SAMPLE = True
TRAINING = False
SIZE_OF_VALIDATION = 100
BATCH_SIZE = 8

data_path_IR = "/home/ferhatcan/Desktop/challengedataset/train/**/*.jpg"
data_path_test_IR = "/home/ferhatcan/Desktop/challengedataset/test/640_flir_hr/*.jpg"


for i in range(len(noise_sigmas)):
    if noise_sigmas[i-1] == noise_sigmas[i]:
        continue

    # Location of model weights (needed for demo)
    weights_dir = f'weights/ircnn_denoise_{noise_sigmas[i] // 1}'
    weights_file = os.path.join(weights_dir, 'weights.h5')
    os.makedirs(weights_dir, exist_ok=True)

    # -----------------------
    #       Dataset
    # -----------------------
    train_ds = IR_DATASET(images_dir=data_path_IR, scale=scale, subset="noise_model",
                          noise_mean=noise_mean, noise_sigma=noise_sigmas[i])

    train_ds = train_ds.dataset(batch_size=16)

    validation_ds = train_ds.take(SIZE_OF_VALIDATION)
    train_ds = train_ds.skip(SIZE_OF_VALIDATION)

    test_ds = IR_DATASET(images_dir=data_path_test_IR, scale=scale, subset="noise_model",
                         noise_mean=noise_mean, noise_sigma=noise_sigmas[0], hr_shape=[200, 200])

    test_ds = test_ds.dataset(batch_size=1, random_transform=False)

    # -----------------------
    #       Trainer
    # -----------------------
    trainer = ircnn_denoise_trainer(model=ircnn_denoise(num_filters=24, noise_level=noise_sigmas[i]),
                                    checkpoint_dir=f'.ckpt/ircnn_denoise_{noise_sigmas[i] // 1}',
                                    learning_rate=PiecewiseConstantDecay(boundaries=[15000], values=[1e-4, 5e-5]))
    if TRAINING:
        trainer.train(train_ds,
                      validation_ds,
                      steps=20000,
                      evaluate_every=200,
                      save_best_only=True)

    # Restore from checkpoint with highest PSNR
    trainer.restore()

    # ------------------------
    #       Evaluation
    # ------------------------
    # Evaluate model on full validation set
    psnrv = trainer.evaluate(test_ds.take(1000))
    print(f'PSNR = {psnrv.numpy():3f} for noise sigma {noise_sigmas[i] // 1}')

    if SHOW_SAMPLE:
        for lr, hr in test_ds.take(10):
            sr = trainer.SR_return(lr)
            show_noise_pair(hr, sr, lr)

    # ------------------------
    #       Save Weights
    # ------------------------
    # Save weights to separate location (needed for demo)
    trainer.model.save_weights(weights_file)



