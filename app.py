import cv2
from PIL import Image
import numpy as np
import os
from keras.models import load_model

run = True
video_name = "person_walking.mp4"
capture = cv2.VideoCapture(video_name)
seg_model = load_model('models/supervisely-unet-model.h5py', compile=False)
ret, frame = capture.read()
frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
height, width, _ = frame.shape
new_video_name = "edited_"+video_name
make_video = cv2.VideoWriter(new_video_name, cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (width, height))

while run:
    ret, frame = capture.read()
    try:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
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
            '256mask.png', mask_img)
        mask_im = cv2.imread(
            '256mask.png')
        mask_im = cv2.cvtColor(mask_im, cv2.COLOR_BGR2RGB)
        mask_im = cv2.resize(mask_im, (width, height))
        mask_im[np.all(mask_im == 1, axis=-1)] = 255
        mask_im = mask_im[:, :, 0]

        print(mask_im.shape)
        human_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask_im)

        frame_copy = cv2.blur(frame_copy, (27, 27))
        background_mask = cv2.bitwise_not(mask_im)
        background_extracted = cv2.bitwise_and(
            frame_copy, frame_copy, mask=background_mask)

        result = cv2.add(human_extracted, background_extracted)
        print(result)
        cv2.imshow('', result)
        make_video.write(result)
         os.removedirs(
            '256mask.png')
        key = cv2.waitKey(10)
        if key == 27:
            break
    except:
        run = False
cv2.destroyAllWindows()
capture.release()

make_video.release()
