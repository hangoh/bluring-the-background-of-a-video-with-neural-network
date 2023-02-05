import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

seg_model = load_model('models/supervisely-unet-model.h5py', compile=False)
vid_cap = cv2.VideoCapture(0)
run = False

if vid_cap.isOpened():
    ret, frame = vid_cap.read()
    if ret:
        run = True
        while run:
            ret, frame = vid_cap.read()
            frame_copy = frame.copy()
            height, width, _ = frame.shape
            print(str(height)+'...'+str(width))
            seg_image = cv2.resize(frame, (256, 256))
            seg_image = Image.fromarray(seg_image)
            seg_image = np.array(seg_image)

            seg_input = np.expand_dims(seg_image, axis=0)
            mask = (seg_model.predict(seg_image.reshape(1, 256, 256, 3)))
            mask_img = np.argmax(mask, axis=3)[0, :, :]

            cv2.imwrite(
                'mask.png', mask_img)
            mask_im = cv2.imread(
                'mask.png')
            mask_im = cv2.cvtColor(mask_im, cv2.COLOR_BGR2RGB)
            mask_im = cv2.resize(mask_im, (width, height))
            mask_im[np.all(mask_im == 1, axis=-1)] = 255
            mask_im = mask_im[:, :, 0]

            print(mask_im.shape)
            human_extracted = cv2.bitwise_and(
                frame_copy, frame_copy, mask=mask_im)

            frame_copy = cv2.blur(frame_copy, (27, 27))
            background_mask = cv2.bitwise_not(mask_im)
            background_extracted = cv2.bitwise_and(
                frame_copy, frame_copy, mask=background_mask)
            print()

            result = cv2.add(human_extracted, background_extracted)

            cv2.imshow('Result', result)
            #run = False
            key = cv2.waitKey(30)
            if key == 27:
                os.remove('mask.png')
                break
        vid_cap.release()
        cv2.distroyAllWindows()
    else:
        print('capture unsuccessful')
else:
    print('no camera found')
