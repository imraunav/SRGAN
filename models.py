import tensorflow as tf
from tensorflow.keras import utils, layers, Model, optimizers, Input, callbacks
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import glob


def pixel_shuffle(scale=2):
        return lambda x: tf.nn.depth_to_space(x, scale)
    
def residual_block(layer_input, filters=64):
    """
    Following specks of the residual block described in paper
    """
    d = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(layer_input)
    d = layers.BatchNormalization()(d)
    d = layers.PReLU(shared_axes=[1, 2])(d)
    d = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(d)
    d = layers.BatchNormalization()(d)
    return layers.Add()([d, layer_input])

def upsampling_block(layer_input):
    u = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(layer_input)
    u = layers.Lambda(pixel_shuffle(scale=2))(u)
    u = layers.PReLU(shared_axes=[1, 2])(u)
    return u
    
def build_generator(lr_shape=(None, None, 3), b_residual_blocks=16):
    """
    Builds a genrator networks according to specs descibed by SRGAN.
    The network takes in a low resolution image and generates a corresponding high resolution image.
    The input to the network is an image of 3 channels. The output is an image of dimention 4x the input image.
    The image tensors are in the range of [0, 255] for both input and output.
    """

    # Low resolution image input
    img_lr = Input(shape=lr_shape)
    # Rescaling to fit the network architecture. Input LR in range [0, 1]
    img_lr = layers.Rescaling(scale=1/255.0)(img_lr) # take care of recaling in network itself

    # Pre-residual block
    c1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(img_lr)
    c1 = layers.PReLU(shared_axes=[1, 2])(c1)

    # 16 residual blocks
    r = residual_block(c1) # first residual block
    for i in range(b_residual_blocks-1): # add residual block one after the other
        r = residual_block(r)

    # Post-residual block
    c2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(r)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Add()([c2, c1]) # skip connection


    # upsampling: the multiplier is controlled here
    u1 = upsampling_block(c2) # x2
    u2 = upsampling_block(u1) # x2

    # Last conv layer
    gen_hr = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same", activation="tanh")(u2)
    # Rescaling to give back image in range [0, 255]
    gen_hr = layers.Rescaling(scale=127.5, offset=127.5)(gen_hr) # maybe this is the correction needed.

    return Model(inputs=[img_lr], outputs=[gen_hr], name="Generator")

def d_block(layer_input, filters=3, strides=1, batchnormalise=True):
    d = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same")(layer_input)
    if batchnormalise == True:
        d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    return d

def build_discriminator(hr_shape=(96,96,3)):
    # Input img
    d0 = Input(shape=hr_shape)
    # d0 = layers.Rescaling(scale=1/127.5, offset=-1)(d0) # normalize data in range [-1, 1]
    d1 = d_block(d0, filters=64, strides=1, batchnormalise=False)
    d2 = d_block(d1, filters=64, strides=2, batchnormalise=True)
    d3 = d_block(d2, filters=128, strides=1, batchnormalise=True)
    d4 = d_block(d3, filters=128, strides=2, batchnormalise=True)
    d5 = d_block(d4, filters=256, strides=1, batchnormalise=True)
    d6 = d_block(d5, filters=256, strides=2, batchnormalise=True)
    d7 = d_block(d6, filters=512, strides=1, batchnormalise=True)
    d8 = d_block(d6, filters=512, strides=2, batchnormalise=True)

    d8_flat = layers.Flatten()(d8)
    d9 = layers.Dense(units=1024)(d8_flat)
    d10 = layers.LeakyReLU(alpha=0.2)(d9)
    score = layers.Dense(units=1, activation="sigmoid")(d10)
    return Model(inputs=[d0], outputs=[score], name="Discriminator")

def VGG_54():
    layer_5_4 = 20
    vgg = VGG19(weights="imagenet", input_shape=(None, None, 3), include_top=False)
    # vgg_54 = Model(inputs=[vgg.input], outputs=[vgg.get_layer('block5_conv4').output])
    vgg_54 = Model(inputs=[vgg.input], outputs=[vgg.layers[layer_5_4].output])

    return vgg_54

class MySRGAN(Model):
    def __init__(self, generator, discriminator, vgg=VGG_54()):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator    
        self.vgg = vgg
        # self.vgg.trainable = False # just for safety

    def compile(self, generator_optimizer, discriminator_optimizer):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

    def content_loss(self, hr, sr):
        sr = (sr + 1) * 127.5 
        hr = (hr + 1) * 127.5
        sr = preprocess_input(sr) # takes in RGB image of range [0, 255]
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr)/12.75
        hr_features = self.vgg(hr)/12.75

        return self.mse(hr_features, sr_features)

    def generator_loss(self, score_sr):
        sr_loss = self.bce(tf.ones_like(score_sr), score_sr)
        return sr_loss

    def discriminator_loss(self, score_hr, score_sr):

        hr_loss = self.bce(tf.ones_like(score_hr), score_hr)
        sr_loss = self.bce(tf.zeros_like(score_sr), score_sr)

        return 0.5*(hr_loss+sr_loss)

    def train_step(self, batch):
        lr = tf.cast(batch[0], tf.float32)
        hr = tf.cast(batch[1], tf.float32)

        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            
            # super-resolve and generate image
            sr = self.generator(lr, training=True)
        
            # Train the discriminator
            score_hr = self.discriminator(hr, training=True)
            score_sr = self.discriminator(sr, training=True)
            
            d_loss = self.discriminator_loss(score_hr, score_sr)
        
            # Train the generator (note that we should *not* update the weights of the discriminator in this step)!
            content_loss = self.content_loss(hr, sr)
            gen_loss = self.generator_loss(score_sr)
            perceptual_loss = content_loss + 0.001 * gen_loss # SRGAN speciality

        # update discriminator weights
        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        # update generator weights
        grads = gtape.gradient(perceptual_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"Discriminator loss":d_loss, "Generator loss":gen_loss, "Perceptual loss": perceptual_loss}
    
    def call(self, latent_vectors):
        # needed to build the network incase of loading the weights
        x = self.generator(latent_vectors)
        return self.discriminator(x)
    

def extract_random_patch(lr_img, hr_img, hr_crop_size=96, scale=4):
    lr_crop_size = hr_crop_size // scale
    lr_shape = tf.shape(lr_img)[:2]
    # print(lr_shape)
    # tf.print(tf.shape(hr_img), output_stream=sys.stdout) # weird way to view the shape of the input tensors

    lr_top = tf.random.uniform(shape=(), maxval=lr_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    lr_left = tf.random.uniform(shape=(), maxval=lr_shape[1] - lr_crop_size + 1, dtype=tf.int32)

    hr_top = lr_top * scale
    hr_left = lr_left * scale

    lr_crop = lr_img[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size]
    hr_crop = hr_img[hr_top:hr_top + hr_crop_size, hr_left:hr_left + hr_crop_size]
    # print(tf.shape(lr_crop))
    return lr_crop, hr_crop

def random_flip(lr_img, hr_img):
    flip_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(flip_chance < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rotate_option = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rotate_option), tf.image.rot90(hr_img, rotate_option)

def random_lr_jpeg_noise(lr_img, hr_img, min_jpeg_quality=50, max_jpeg_quality=95):
    jpeg_noise_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(jpeg_noise_chance < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.random_jpeg_quality(lr_img, min_jpeg_quality, max_jpeg_quality),
                            hr_img))


def create_patch_dataset(lr_path, hr_path):
    lr_filenames = sorted(glob.glob(lr_path+"/*.png"))#sorted(os.listdir(path))
    hr_filenames = sorted(glob.glob(hr_path+"/*.png"))#sorted(os.listdir(path))

    lr_ds = []
    hr_ds = []

    for lr_filename, hr_filename in zip(lr_filenames, hr_filenames):
        lr = tf.io.read_file(lr_filename)
        lr = tf.image.decode_png(lr, channels=3)

        hr = tf.io.read_file(hr_filename)
        hr = tf.image.decode_png(hr, channels=3)

        for _ in range(1): # extract 1 patches
            lr_patch, hr_patch = extract_random_patch(lr, hr, hr_crop_size=96, scale=4)
            lr_ds.append(lr_patch)
            hr_ds.append(hr_patch)
    lr_ds = tf.data.Dataset.from_tensor_slices(lr_ds)
    hr_ds = tf.data.Dataset.from_tensor_slices(hr_ds)
    dataset = tf.data.Dataset.zip((lr_ds, hr_ds))
    return dataset


def dataset_preprocess():
    # hr_size = 1000
    path_hr = "./DIV2K_hr/DIV2K_train_HR/DIV2K_train_HR"
    path_lr = "./DIV2K_train_LR_bicubic/X4"

    # Read image pairs and extract patches
    dataset = create_patch_dataset(path_lr, path_hr)

    # Apply random Rotation
    dataset = dataset.map(random_rotate)


    # Apply random flip
    dataset = dataset.map(random_flip)

    # Add JPEG Noise
    dataset = dataset.map(lambda lr, hr: random_lr_jpeg_noise(lr, hr))

    # View dataset for sanity
    # view_dataset(dataset)
    return dataset



# old training step
# with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        #     # super-resolve and generate image
        #     sr = self.generator(lr, training=True)
        #     all_images = tf.concat([hr, sr], axis=0)

        #     #Assemble labels(reals are labeled 1, fakes are labeled 0)
        #     labels = tf.concat([
        #         tf.ones((tf.shape(hr)[0],)), 
        #         tf.zeros((tf.shape(sr)[0],)),
        #         ], axis=0)
        
        #     # Train the discriminator
        #     score = self.discriminator(all_images, training=True)
        #     d_loss = self.bce(labels, score)
            
        #     # Assemble labels that say "all real images"
        #     misleading_labels = tf.ones((tf.shape(sr)[0], 1))
        
        #     # Train the generator (note that we should *not* update the weights of the discriminator in this step)!
        #     content_loss = self.content_loss(hr, sr)
        #     score = self.discriminator(sr, training=False)
        #     gen_loss = self.bce(misleading_labels, score)
        #     perceptual_loss = content_loss + 0.001 * gen_loss # SRGAN speciality  