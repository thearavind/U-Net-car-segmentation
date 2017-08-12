import cv2
import numpy as np
import os
from PIL import Image
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage.io import imsave
from tqdm import tqdm

TRAIN_DIR = 'project-folder/train'
MASK_DIR = 'project-folder/train_jpeg_masks'
GIF_MASK_DIR = 'project-folder/train_masks'
IMG_SIZE = 192
smooth = 1.


def create_train_data():
    training_data = np.ndarray((31, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    i = 0
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if i <= 30:
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data[i] = np.array([img])
            i = i + 1
    np.save('train_data.npy', training_data)


def create_dummy_test_data():
    training_data = np.ndarray((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    i = 0
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if i == 355:
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data[0] = np.array([img])
        i = i + 1
    np.save('dummy_test_data.npy', training_data)


def create_train_mask_data():
    training_data = np.ndarray((31, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    i = 0
    for img in tqdm(os.listdir(MASK_DIR)):
        if i <= 30:
            path = os.path.join(MASK_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data[i] = np.array([img])
        i = i + 1
    np.save('train_mask_data.npy', training_data)


def gif_to_jpeg():
    for img in tqdm(os.listdir(GIF_MASK_DIR)):
        path = os.path.join(GIF_MASK_DIR, img)
        p = Image.open(path)
        i = p.convert("RGB")
        i.save("train_jpeg_masks/" + img.split('.')[0] + ".jpg", "JPEG", quality=80, optimize=True, progressive=True)


def load_npy_file():
    return np.load('train_data.npy').astype('float32'), np.load('train_mask_data.npy').astype('float32')


def load_dummy_npy_file():
    return np.load('dummy_test_data.npy').astype('float32')


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


gif_to_jpeg()
create_train_data()
create_train_mask_data()

npy_image, npy_mask = load_npy_file()
npy_mask = npy_mask[..., np.newaxis]

mean = np.mean(npy_image)
std = np.std(npy_image)

npy_image -= mean
npy_image /= std

npy_mask /= 255.

model = get_unet()

model_checkpoint = ModelCheckpoint('weights_checkpoint.h5', monitor='val_loss', save_best_only=True)

model.fit(npy_image, npy_mask, batch_size=32, nb_epoch=20, shuffle=True,
          validation_split=0.2, callbacks=[model_checkpoint])

create_dummy_test_data()
model.load_weights('weights_checkpoint.h5')
test_data = load_dummy_npy_file()
imgs_mask_test = model.predict(test_data)
np.save('imgs_mask_test.npy', imgs_mask_test)

pred_dir = 'preds'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

for image in np.load('imgs_mask_test.npy'):
    print('Image', image)
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join('prediction.png'), image)
