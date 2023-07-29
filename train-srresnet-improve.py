from models import build_discriminator, build_generator, apply_random_crop, random_flip, random_rotate, random_lr_jpeg_noise

import tensorflow as tf
from tensorflow.keras import preprocessing, utils, optimizers, callbacks, losses
from matplotlib import pyplot as plt
import os
import numpy as np


class EpochProgress(callbacks.Callback):
    def __init__(self, per_epoch, *args, **kargs):
        super().__init__(*args, **kargs)
        test_img_path = "./track-progress/img/LR/baboon.png"
        lowres_img = utils.load_img(test_img_path)
        lowres_img = utils.img_to_array(lowres_img)
        lowres_img = process_lr(lowres_img)
        self.test_img = lowres_img
        self.per_epoch = per_epoch
        self.loss = []

    # def on_test_batch_end(self, batch, logs=None):
    #     self.psnr.append(10 * np.log10(1 / logs["loss"]))

    # def on_epoch_begin(self, epoch, logs=None):
    #     self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        # print(f"Mean PSNR for epoch: {tf.reduce_mean(self.psnr): .3f}")
        # logs here are the instantantaneous log not the accumulated log
        self.loss.append(logs["loss"])
        plt.plot(self.loss)
        plt.legend(["SRResNet loss"])
        plt.savefig("srresnet-loss-div2k-plot-improve.png")
        plt.close()

        if (epoch+1) % self.per_epoch == 0 or epoch < 100:
            prediction = self.model.predict(tf.expand_dims(self.test_img, axis=0), verbose=0)
            # plt.imsave(f"epoch-{epoch}.png", tf.squeeze(prediction, axis=0))
            # prediction = process_hr(prediction)
            utils.array_to_img(tf.squeeze(prediction, axis=0)).save(f"./track-progress/srresnet-improve/epoch-{epoch+1}.png")
        
        with open("./track-progress/srresnet-improve/current-epoch", mode="w") as file:
            file.write(f"Epoch : {epoch+1}")

def process_hr(img):
    return (tf.cast(img, tf.float64)/127.5) - 1
def process_lr(img):
    return tf.cast(img, tf.float64)/255.0

def main():
    hr_path = "./DIV2K_hr/DIV2K_train_HR/"
    lr_path = "./DIV2K_train_LR_bicubic/"
    
    # hyperparameters
    batch_size = 16
    upscale_factor = 4
    crop_size = 96
    image_size = 1000
    learning_rate = 1e-8
    epochs =  2000 # 1_000_000/(800/16) # prescribed epochs

    train_hr = preprocessing.image_dataset_from_directory(
        directory = hr_path,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=False,
        seed=1327,
        interpolation="nearest", # necessary to avoid applying any transformation on data
        crop_to_aspect_ratio=True,
        label_mode=None
    )

    train_lr = preprocessing.image_dataset_from_directory(
        directory = lr_path,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size//upscale_factor, image_size//upscale_factor),
        shuffle=False,
        seed=1327,
        interpolation="nearest",
        crop_to_aspect_ratio=True,
        label_mode=None
    )

    train_hr = train_hr.map(process_hr)
    train_lr = train_lr.map(process_lr)
    # train_ds = tf.data.Dataset.from_tensor_slices((train_lr, train_hr))
    train_ds = tf.data.Dataset.zip((train_lr, train_hr))
    train_ds = train_ds.map(lambda lr, hr: apply_random_crop(lr, hr, crop_size, upscale_factor))
    train_ds = train_ds.map(random_flip)
    train_ds = train_ds.map(random_rotate)
    # train_ds = train_ds.map(random_lr_jpeg_noise)
    train_ds = train_ds.prefetch(buffer_size=batch_size)

    mseLoss = losses.MeanSquaredError()
    adam = optimizers.Adam(learning_rate=learning_rate) 
    # build the model
    SRResNet = build_generator()
    # SRResNet.load_weights("./")
    SRResNet.load_weights("./models/srresnet-pretrianed.h5") # load pretrained weights

    SRResNet.compile(loss=mseLoss, optimizer=adam)

    log = SRResNet.fit(train_ds, epochs=epochs, verbose=2, callbacks=[EpochProgress(per_epoch=1000)])

    SRResNet.save_weights("./models/srresnet-improve-pretrianed.h5")


    # img = utils.load_img("./track-progress/img/LR/baboon.png")
    # img_arr = utils.img_to_array(img)
    # print(img_arr.shape)
    # out = SRResNet.predict(tf.expand_dims(img_arr, axis=0))
    # print(out.max(), out.min())
    # img.save("lr.png")
    # utils.array_to_img(tf.squeeze(out, axis=0)).save("sr_mse-div2k.png")


    # plt.plot(log.history["loss"])
    # plt.legend(["SRResNet-div2k loss"])
    # plt.savefig("srresnet-loss-div2k-plot.png")

    return None

if __name__ == "__main__":
    main()
