import os
import numpy as np
from sklearn.utils import shuffle

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import tensorflow_addons as tfa
import datetime

import cv2

# Set some parameters
IMG_SHAPE = 512
IMG_CHANNELS = 1
BATCH_SIZE = 3
SPLIT = 15
EPOCHS = 80
AUTO = tf.data.experimental.AUTOTUNE

TRAIN_PATH =      './dots/train/image'
TEST_PATH =       './dots/test/'
ANNOTATION_PATH = './dots/train/label/'

input_img_paths = sorted(
    [
        os.path.join(TRAIN_PATH, fname)
        for fname in os.listdir(TRAIN_PATH)
        if fname.endswith(".jpg")
    ]
)
annotation_img_paths = sorted(
    [
        os.path.join(ANNOTATION_PATH, fname)
        for fname in os.listdir(ANNOTATION_PATH)
        if fname.endswith(".jpg") and not fname.startswith(".")
    ]
)
test_img_paths = sorted(
    [
        os.path.join(TEST_PATH, fname)
        for fname in os.listdir(TEST_PATH)
        if fname.endswith(".jpg")
    ]
)

# Data loader and augmentation
def load_image(img_filepath, mask_filepath, rotate=0, Hflip=False, Vflip=False, brightness=0, zoom=0, contrast=0):
    # adapatation from https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    img = img_orig = tf.io.read_file(img_filepath)
    img = tf.io.decode_jpeg(img, channels=IMG_CHANNELS)
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])

    mask = mask_orig = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SHAPE, IMG_SHAPE])
    
    # zoom in a bit
    if tf.random.uniform(()) > 0.5 and zoom != 0:
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

    # normalize
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

input_img_paths, annotation_img_paths = shuffle(input_img_paths, annotation_img_paths, random_state=42)
input_img_paths_train, annotation_img_paths_train = input_img_paths[: -SPLIT], annotation_img_paths[: -SPLIT]
input_img_paths_test, annotation_img_paths_test = input_img_paths[-SPLIT:], annotation_img_paths[-SPLIT:]

trainloader = tf.data.Dataset.from_tensor_slices((input_img_paths_train, annotation_img_paths_train))
testloader = tf.data.Dataset.from_tensor_slices((input_img_paths_test, annotation_img_paths_test))

trainloader = (
    trainloader
    .shuffle(10)
    .map(lambda x, y: load_image(x, y, rotate=0))
    # .map(load_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

testloader = (
    testloader
    .map(load_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Build U-Net model

def get_model_Unet_v1():

    kernel_size=9

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

def get_model_Unet_v2(kernel_size):

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

# Callbacks

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "E60_no_rot"
image_writer = tf.summary.create_file_writer(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, write_graph=True)
earlystopper = EarlyStopping(patience=10, verbose=2, min_delta=0.01, monitor="loss")
checkpointer = ModelCheckpoint('model-checkpoint.h5', verbose=0, save_best_only=True)

# Create model
model = get_model_Unet_v2(21)

# Compile model 1:
# Buena separacion con el fondo, reconocimiento completo de la zona
# Unet_v1, IMG_SHAPE = 512, IMG_CHANNELS = 1, BATCH_SIZE = 4, SPLIT = 15, EPOCHS = 50
# optimizer='sgd', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy()]

# Compile model 2:
# Buena separacion con el fondo, reconocimiento completo de la zona
# Unet_v2, IMG_SHAPE = 512, IMG_CHANNELS = 1, BATCH_SIZE = 3, SPLIT = 15, EPOCHS = 60
# optimizer='sgd', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy()]
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryCrossentropy()])

# model.summary()

# Fit model

fit = True
if fit:
    results = model.fit(trainloader, epochs=EPOCHS, callbacks=[checkpointer, tensorboard_callback, earlystopper])
model = load_model('model-checkpoint.h5')

# Predict on train, val and test

test_img, test_mask = next(iter(testloader))
pred_mask = model.predict(test_img)

pred_mask_n = pred_mask
pred_mask_n *= (255.0/pred_mask.max())
pred_mask_n = pred_mask_n.astype(np.uint8)

# Plot and print for evaluation

test_img_ndarray = test_img.numpy()
test_mask_ndarray = test_mask.numpy()

with image_writer.as_default():
    tf.summary.image("Validation image", test_img[:4], step=0)
    tf.summary.image("Validation mask", test_mask[:4], step=0)
    tf.summary.image("Predicted masks", pred_mask[:4], step=0)
