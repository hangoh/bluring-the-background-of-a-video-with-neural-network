import natsort
import numpy as np
import os
from keras.models import load_model
import random
import cv2
from PIL import Image
from matplotlib import pyplot as plt

model = load_model('models/supervisely-unet-model.h5py', compile=False)

test_image_path = '/Users/gohyuhan/Downloads/archive (4)/test_image'


def load_random_test_file(path):
    all_file = natsort.natsorted(os.listdir(path), reverse=False)
    random_num = random.randint(0, int(len(all_file)-1))
    print(all_file[random_num])
    image = cv2.imread(path+'/'+all_file[4])
    ori_height = image.shape[0]
    ori_width = image.shape[1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (256, 256))
    image = Image.fromarray(image)
    image = np.array(image)
    return image, ori_height, ori_width


test_image, h, w = load_random_test_file(test_image_path)
print(h)
test_img_input = np.expand_dims(test_image, axis=0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 6))
plt.imshow(predicted_img)
plt.show()


cv2.imwrite(
    '/Users/gohyuhan/Downloads/instance-level_human_parsing/Training/256mask.png', predicted_img)
im = cv2.imread(
    '/Users/gohyuhan/Downloads/instance-level_human_parsing/Training/256mask.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#im = cv2.resize(im, (w, h))
cv2.imwrite(
    '/Users/gohyuhan/Downloads/instance-level_human_parsing/Training/256mask.png', im)
