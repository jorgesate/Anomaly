import os
import numpy as np
from sklearn.utils import shuffle

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import glob

import cv2

# Set some parameters
IMG_SHAPE = 512                 # Image size
IMG_CHANNELS = 3                # Channels of the original image
BATCH_SIZE = 3                  # Image to pass at once to the net, RAM usage 
SPLIT = 30                      # Size in % of the Train/Val split
EPOCHS = 60                     # Cicles for training

ELEMENT =         'bottle'
TRAIN_PATH =      './dataset/' + ELEMENT + '/test/*/*' 
ANNOTATION_PATH = './dataset/' + ELEMENT + '/ground_truth/*/*' 
TEST_PATH = './dataset/' + ELEMENT + '/test/*/*'
AUTO = tf.data.experimental.AUTOTUNE

input_img_paths = sorted(
    [
        os.path.join(fname) for fname in glob.glob(TRAIN_PATH) if fname.endswith(".png")
    ])
annotation_img_paths = sorted(
    [
        os.path.join(fname) for fname in glob.glob(ANNOTATION_PATH) if fname.endswith(".png")
    ])

N_SPLIT = int(len(input_img_paths)*SPLIT/100)
input_img_paths, annotation_img_paths = shuffle(input_img_paths, annotation_img_paths, random_state=42)
input_img_paths_train, annotation_img_paths_train = input_img_paths[: -N_SPLIT], annotation_img_paths[: -N_SPLIT]
input_img_paths_val, annotation_img_paths_val = input_img_paths[-N_SPLIT:], annotation_img_paths[-N_SPLIT:]

trainloader = tf.data.Dataset.from_tensor_slices((input_img_paths_train, annotation_img_paths_train))
valLoader = tf.data.Dataset.from_tensor_slices((input_img_paths_val, annotation_img_paths_val))

# Data loader and augmentation
def load_image(img_filepath, mask_filepath, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0):
    # adapatation from https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    img = img_orig = tf.io.read_file(img_filepath)
    img = tf.io.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])

    mask = mask_orig = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.image.resize(mask, [IMG_SHAPE, IMG_SHAPE])
    
    # zoom in a bit
    if zoom != 0 and (tf.random.uniform(()) > 0.5):

        # use original image to preserve high resolution
        img = tf.image.central_crop(img, zoom)
        mask = tf.image.central_crop(mask, zoom)
        # resize
        img = tf.image.resize(img, (IMG_SHAPE, IMG_SHAPE))
        mask = tf.image.resize(mask, (IMG_SHAPE, IMG_SHAPE))
    
    # random brightness adjustment illumination
    if brightness != 0:
        img = tf.image.random_brightness(img, brightness)
    # random contrast adjustment
    if contrast != 0:
        img = tf.image.random_contrast(img, 1-contrast, 1+2*contrast)
    
    # flipping random horizontal 
    if tf.random.uniform(()) > 0.5 and Hflip:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    # or vertical
    if tf.random.uniform(()) > 0.5 and Vflip:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # rotation in 360° steps
    if rotate != 0:
        rot_factor = tf.cast(tf.random.uniform(shape=[], minval=-rotate, maxval=rotate, dtype=tf.int32), tf.float32)
        angle = np.pi/360*rot_factor
        img = tfa.image.rotate(img, angle)
        mask = tfa.image.rotate(mask, angle)

    return img, mask

trainloader = (
    trainloader
    .shuffle(len(input_img_paths))
    .map(lambda x, y: load_image(x, y, rotate=10, Hflip=True, Vflip=True, brightness=0.1, contrast=0.1), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

valLoader = (
    valLoader
    .map(lambda x, y: load_image(x, y), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))

# Build U-Net model

def get_model_Unet_v1(kernel_size):

    kernel_size=kernel_size

    inputs = Input((IMG_SHAPE, IMG_SHAPE, IMG_CHANNELS))

    c1 = Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D(2) (c1)

    c2 = Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D(2) (c2)

    c3 = Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D(2) (c3)

    c4 = Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=2) (c4)

    c5 = Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    return Model(inputs, outputs)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x
def get_model_Unet_v2(input_img, n_filters = 16, dropout = 0.1, kernel_size = 3, batchnorm = True):
    """Function to define the UNET Model"""

    input = Input(input_img, name='img')

    # Contracting Path
    c1 = conv2d_block(input, n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = kernel_size, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = kernel_size, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = kernel_size, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, kernel_size = kernel_size, strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = kernel_size, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, kernel_size = kernel_size, strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = kernel_size, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, kernel_size = kernel_size, strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, kernel_size = kernel_size, strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(input, outputs)

# Callbacks

str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + str_time
img_dir = "logs/" + str_time + "/img"
model_name = 'model-checkpoint_' + ELEMENT + '.h5'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True)
earlystopper = EarlyStopping(patience=10, verbose=2, min_delta=0.01, monitor="loss")
checkpointer = ModelCheckpoint(model_name, verbose=0, save_best_only=True)

# Create model
model = get_model_Unet_v2((IMG_SHAPE, IMG_SHAPE, 1), n_filters=32, dropout=0.2, kernel_size = 3, batchnorm=True)

# Compile model 1:
# Buena separacion con el fondo, reconocimiento completo de la zona
# Unet_v1, IMG_SHAPE = 512, IMG_CHANNELS = 1, BATCH_SIZE = 4, SPLIT = 15, EPOCHS = 50
# optimizer='sgd', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy()]
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])

# model.summary()

# Fit model

fit = True
if fit:
    results = model.fit(trainloader, epochs=EPOCHS, validation_data=valLoader, callbacks=[checkpointer, tensorboard_callback])
model = load_model(model_name)

# Predict on train, val and test

val_img, val_mask = next(iter(valLoader))
pred_mask = model.predict(val_img)

# Plot and print for evaluation

val_img_ndarray = val_img.numpy()
val_mask_ndarray = val_mask.numpy()

image_writer = tf.summary.create_file_writer(img_dir)
with image_writer.as_default():
    tf.summary.image("Validation image", val_img[:4], step=0)
    tf.summary.image("Validation mask", val_mask[:4], step=0)
    tf.summary.image("Predicted masks", pred_mask[:4], step=0)
