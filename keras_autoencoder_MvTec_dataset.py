import os
import numpy as np
from sklearn.utils import shuffle

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, LeakyReLU, Flatten, Reshape
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import glob

import matplotlib.pyplot as plt

import cv2

import skimage

# Set some parameters
IMG_SHAPE = 256                 # Image size
IMG_CHANNELS = 3                # Channels of the original image
BATCH_SIZE = 3                  # Image to pass at once to the net, RAM usage 
SPLIT = 30                      # Size in % of the Train/Val split
EPOCHS = 15                     # Cicles for training
PATIENCE = 9
MIN_DELTA = 0.1
TRAIN = False

ELEMENT =           'bottle'
TRAIN_PATH =        './dataset/' + ELEMENT + '/train/*/*' 
TEST_PATH =         './dataset/' + ELEMENT + '/test/*/*'
AUTO = tf.data.experimental.AUTOTUNE

input_img_paths = sorted(
    [
        os.path.join(fname) for fname in glob.glob(TRAIN_PATH) if fname.endswith(".png")
    ])

test_img_paths = sorted(
    [
        os.path.join(fname) for fname in glob.glob(TEST_PATH) if fname.endswith(".png")
    ])

N_SPLIT = int(len(input_img_paths)*SPLIT/100)
input_img_paths = shuffle(input_img_paths, random_state=42)
input_img_paths_train = input_img_paths[: -N_SPLIT]
input_img_paths_val = input_img_paths[-N_SPLIT:]

print('\nSe usaran {} imagenes para el entrenamiento y {} para la validación\n'.format(len(input_img_paths_train), len(input_img_paths_val)))

trainloader = tf.data.Dataset.from_tensor_slices((input_img_paths_train, input_img_paths_train))
valLoader = tf.data.Dataset.from_tensor_slices((input_img_paths_val, input_img_paths_val))
testLoader = tf.data.Dataset.from_tensor_slices((test_img_paths, test_img_paths))

# Data loader and augmentation
def load_image(img_filepath, y, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0):
    # adapatation from https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    img = img_orig = tf.io.read_file(img_filepath)
    img = tf.io.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])
    
    # zoom in a bit
    if zoom != 0 and (tf.random.uniform(()) > 0.5):

        # use original image to preserve high resolution
        img = tf.image.central_crop(img, zoom)
        # resize
        img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE))
    
    # random brightness adjustment illumination
    if brightness != 0:
        img = tf.image.random_brightness(img, brightness)
    # random contrast adjustment
    if contrast != 0:
        img = tf.image.random_contrast(img, 1-contrast, 1+2*contrast)
    
    # flipping random horizontal 
    if tf.random.uniform(()) > 0.5 and Hflip:
        img = tf.image.flip_left_right(img)
    # or vertical
    if tf.random.uniform(()) > 0.5 and Vflip:
        img = tf.image.flip_up_down(img)

    # rotation in 360° steps
    if rotate != 0:
        rot_factor = tf.cast(tf.random.uniform(shape=[], minval=-rotate, maxval=rotate, dtype=tf.int32), tf.float32)
        angle = np.pi/360*rot_factor
        img = tfa.image.rotate(img, angle)

    y = img

    return img, y

trainloader = (
    trainloader
    .shuffle(len(input_img_paths_train))
    .map(lambda x, y: load_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valLoader = (
    valLoader
    .map(lambda x, y: load_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

testLoader = (
    testLoader
    .map(lambda x, y: load_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

# Build U-Net model

def build(inputShape, filters=(32, 64), latentDim=16):
    # initialize the input shape to be "channels last" along with
    # the channels dimension itself
    # channels dimension itself
    chanDim = -1
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs
    # loop over the number of filters
    for f in filters:
        # apply a CONV => RELU => BN operation
        x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)
    # flatten the network and then construct our latent vector
    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)
    # build the encoder model
    encoder = Model(inputs, latent, name="encoder")
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latentInputs = Input(shape=(latentDim,))
    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=chanDim)(x)
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    x = Conv2DTranspose(inputShape[2], (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)
    # build the decoder model
    decoder = Model(latentInputs, outputs, name="decoder")
    # our autoencoder is the encoder + decoder
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return (encoder, decoder, autoencoder)

# Callbacks

str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + str_time + "_AE"
img_dir = "logs/" + str_time + "_AE" + "/img"
model_name = 'saves/mc_AE_' + ELEMENT + '.h5'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True)
earlystopper = EarlyStopping(patience=PATIENCE, verbose=2, min_delta=MIN_DELTA, monitor="loss")
checkpointer = ModelCheckpoint(model_name, verbose=0, save_best_only=True)

# Create model
# model = get_model_Unet_v1((IMG_SHAPE, IMG_SHAPE, 1), n_filters=32, dropout=0.2, kernel_size = 3, batchnorm=True)

(encoder, decoder, autoencoder) = build((IMG_SHAPE, IMG_SHAPE, 1))

# Compile model 1:
autoencoder.compile(optimizer='Adam', loss='mse', metrics=["accuracy"])

# model.summary()

# Fit model

if TRAIN:
    results = autoencoder.fit(trainloader, validation_data=valLoader, epochs=EPOCHS, callbacks=[checkpointer, earlystopper])

model = load_model(model_name)

# Predict on train, val and test

print("\nEvaluating...\n")

# val_img, _ = next(iter(valLoader))
# pred_val_img = model.predict(val_img)

test_img, _ = next(iter(testLoader))
pred_test_img = model.predict(test_img)

# Plot and print for evaluation

# val_img_ndarray = val_img.numpy()
# val_img_ndarray = val_img_ndarray.astype("uint8")

test_img_ndarray = test_img.numpy()
# test_img_ndarray = test_img_ndarray.astype("uint8")
# pred_test_img = pred_test_img.astype("uint8")

diff_array = []

f, axs = plt.subplots(2, 2)

for i in range(BATCH_SIZE):

    pti = pred_test_img[i] * 255
    pti = pti.astype("uint8")
    tin = test_img_ndarray[i] * 255
    tin = tin.astype("uint8")
    diff = cv2.absdiff(pti, tin)

    axs[0, 0].imshow(pti, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(tin, cmap='gray')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(diff, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(diff, cmap='gray')
    axs[1, 1].axis('off')

    plt.waitforbuttonpress(2 )

plt.close()

'''with image_writer.as_default():
    tf.summary.image("Validation image", val_img[:BATCH_SIZE], step=0)
    tf.summary.image("Predicted image", pred_val_img[:BATCH_SIZE], step=0)'''