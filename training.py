from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import pickle
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,  EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
import natsort
from Unet import multi_unet_model, jacard_coef
import segmentation_models as sm

# set device to gpu if have one
with tf.device('/GPU'):
    a = tf.random.normal(shape=(2,), dtype=tf.float32)
    b = tf.nn.relu(a)


def data_loader(folder_dir):
    image_dataset = []
    print('start...')
    num = 0
    for images in natsort.natsorted(os.listdir(folder_dir), reverse=False):
        if num < 4000:
            image = cv2.imread(folder_dir+'/'+images)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            image = Image.fromarray(image)
            image = np.array(image)
            image_dataset.append(image)
            num = num+1
        else:
            break
    print('end...')
    return image_dataset


image_dataset = data_loader(
    "{{Enter your RGB dataset path here}}")

mask_dataset = data_loader(
    "{{Enter your label dataset path here}}")


image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)


print(mask_dataset.shape)

# Sanity check...
'''
image_number = 0
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(mask_dataset[image_number])
plt.show()
'''


def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg[np.all(label == (0, 0, 0), axis=-1)] = 0
    label_seg[np.all(label == (255, 255, 255), axis=-1)] = 1

    # Just take the first channel, no need for all 3 channels
    label_seg = label_seg[:, :, 0]

    return label_seg


labels = []

print('hot encoding start')
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)


print('hot encoding end')
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

# Another Sanity check...
'''
image_number = 0
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number])
plt.show()
'''
print('to_categorical function start')
n_classes = len(np.unique(labels))
labels_categories = to_categorical(labels, num_classes=n_classes)
print('to_categorical function end')

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, labels_categories, test_size=0.02, random_state=42)

weights = compute_class_weight(class_weight='balanced', classes=np.unique(
    np.ravel(labels, order='C')), y=np.ravel(labels, order='C'))

print(weights)
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

history = model.fit(image_dataset, labels_categories,
                    batch_size=10,
                    verbose=1,
                    epochs=15,
                    validation_data=(X_test, y_test),
                    shuffle=False)

model.save('models/{{enter desire name for your model}}.h5py')
