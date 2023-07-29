from models import build_discriminator, build_generator,  dataset_preprocess, MySRGAN

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
        self.dloss = []
        # self.gloss = []
        self.ploss = []

    # def on_test_batch_end(self, batch, logs=None):
    #     self.psnr.append(10 * np.log10(1 / logs["loss"]))

    # def on_epoch_begin(self, epoch, logs=None):
    #     self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        # print(f"Mean PSNR for epoch: {tf.reduce_mean(self.psnr): .3f}")
        # logs here are the instantantaneous log not the accumulated log

        self.dloss.append(logs["Discriminator loss"])
        # self.gloss.append(logs["Generator loss"])
        self.ploss.append(logs["Perceptual loss"])

        plt.plot(self.dloss, label="Discriminator loss")
        # plt.plot(self.gloss, label="Generator loss")
        plt.plot(self.ploss, label="Perceptual loss")
        plt.legend()
        plt.savefig("./track-progress/srgan/srgan-loss-div2k-plot.png")
        plt.close()

        # if (epoch+1) % self.per_epoch == 0 or epoch < 100:
            # prediction = self.model.generator.predict(tf.expand_dims(self.test_img, axis=0), verbose=0)
            # # plt.imsave(f"epoch-{epoch}.png", tf.squeeze(prediction, axis=0))
            # # prediction = process_hr(prediction)
            # utils.array_to_img(tf.squeeze(prediction, axis=0)).save(f"./track-progress/srgan/epoch-{epoch+1}.png")
        
        prediction = self.model.generator.predict(tf.expand_dims(self.test_img, axis=0), verbose=0)
        utils.array_to_img(tf.squeeze(prediction, axis=0)).save(f"./track-progress/srgan/epoch-progress.png")
    
        with open("./track-progress/srgan/current-epoch", mode="w") as file:
            file.write(f"Epoch : {epoch+1}")

def process_hr(img):
    return (tf.cast(img, tf.float32)/127.5) - 1
def process_lr(img):
    return tf.cast(img, tf.float32)/255.0


def main():
    # hr_path = "./DIV2K_hr/DIV2K_train_HR/"
    # lr_path = "./DIV2K_train_LR_bicubic/"
    
    # hyperparameters
    batch_size = 16
    upscale_factor = 4
    crop_size = 96
    image_size = 1000
    learning_rate = optimizers.schedules.PiecewiseConstantDecay(boundaries=[100_000], values=[1e-4, 1e-5]) # following paper huristics
    training_steps = 2_00_000
    steps_per_epoch = 1000
    epochs = training_steps // steps_per_epoch # 2*100_000/(1600/16) # more samples in dataset

    train_ds = dataset_preprocess()
    train_ds = train_ds.map(lambda lr, hr: (process_lr(lr), process_hr(hr)))

    train_ds = train_ds.batch(batch_size=batch_size)
    train_ds = train_ds.repeat() # repeat the dataset indefinitely, control the rest with .fit method
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # building the model

    generator = build_generator()
    discriminator = build_discriminator()

    # generator.load_weights("./models/srresnet-pretrianed.h5") # load pretrained weights
    generator.load_weights("./models/downloaded/SRResNet.h5")

    srgan = MySRGAN(generator, discriminator)

    gen_opt = optimizers.Adam(learning_rate=learning_rate)
    dis_opt = optimizers.Adam(learning_rate=learning_rate)

    srgan.compile(gen_opt, dis_opt)

    log = srgan.fit(train_ds, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch,
                    callbacks=[
                        EpochProgress(per_epoch=1000),
                        callbacks.ModelCheckpoint(
                            filepath="./checkpoints/srgan-model.h5",
                            save_weights_only=True,
                            save_best_only=False, save_freq="epoch"),
                    ])
    filepath = "./models"
    srgan.generator.save_weights(os.path.join(filepath,"./downloaded/trained/srgan-generator.h5"), overwrite=True, save_format=None, options=None)
    srgan.generator.save_weights(os.path.join(filepath,"./downloaded/trained/srgan-discriminator.h5"), overwrite=True, save_format=None, options=None)





if __name__ == "__main__":
    main()