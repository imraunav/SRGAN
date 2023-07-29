from models import build_generator, dataset_preprocess

import tensorflow as tf
from tensorflow.keras import preprocessing, utils, optimizers, callbacks, losses
from matplotlib import pyplot as plt
import os
import numpy as np


class EpochProgress(callbacks.Callback):
    def __init__(self, per_epoch=1000, *args, **kargs):
        super().__init__(*args, **kargs)
        test_img_path = "./track-progress/img/LR/baboon.png"
        lowres_img = utils.load_img(test_img_path)
        lowres_img = utils.img_to_array(lowres_img)
        # lowres_img = process_lr(lowres_img)
        self.test_img = lowres_img
        self.per_epoch = per_epoch
        self.loss = []

    def on_epoch_end(self, epoch, logs=None):
        # logs here are the instantantaneous log not the accumulated log
        self.loss.append(logs["loss"])
        plt.plot(self.loss)
        plt.legend(["SRResNet loss"])
        plt.savefig("./track-progress/srresnet/srresnet-loss-div2k-plot.png")
        plt.close()

        # if (epoch+1) % self.per_epoch == 0 or epoch < 100:
        #     prediction = self.model.predict(tf.expand_dims(self.test_img, axis=0), verbose=0)
        #     # plt.imsave(f"epoch-{epoch}.png", tf.squeeze(prediction, axis=0))
        #     # prediction = process_hr(prediction)
        #     utils.array_to_img(tf.squeeze(prediction, axis=0)).save(f"./track-progress/srresnet/epoch-{epoch+1}.png")
        
        prediction = self.model.predict(tf.expand_dims(self.test_img, axis=0), verbose=0)
        utils.array_to_img(tf.squeeze(prediction, axis=0)).save(f"./track-progress/srresnet/epoch-progress.png")
        
        with open("./track-progress/srresnet/current-epoch", mode="w") as file:
            file.write(f"Epoch : {epoch+1}")

def process_hr(img):
    return (tf.cast(img, tf.float64)/127.5) - 1
def process_lr(img):
    return tf.cast(img, tf.float64)/255.0


def main():
    # hr_path = "./DIV2K_hr/DIV2K_train_HR/"
    # lr_path = "./DIV2K_train_LR_bicubic/"
    
    # hyperparameters
    batch_size = 16
    upscale_factor = 4
    crop_size = 96
    # image_size = 1000
    learning_rate = 1e-4
    # epochs =  20_000 # 1_000_000/(800/16) # prescribed epochs
    training_steps = 1_000_000
    steps_per_epoch = 1000
    epochs = training_steps // steps_per_epoch # 1_000_000/(1600/16) # more samples in dataset

    train_ds = dataset_preprocess()
    # train_ds = train_ds.map(lambda lr, hr: (process_lr(lr), process_hr(hr)))
    train_ds = train_ds.batch(batch_size=batch_size)
    train_ds = train_ds.repeat() # repeat the dataset indefinitely, control the rest with .fit method
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    mseLoss = losses.MeanSquaredError()
    adam = optimizers.Adam(learning_rate=learning_rate) 
    # build the model
    SRResNet = build_generator()

    SRResNet.compile(loss=mseLoss, optimizer=adam)

    log = SRResNet.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[EpochProgress()])

    SRResNet.save_weights("./models/srresnet-pretrianed.h5")

    return None

if __name__ == "__main__":
    main()
