# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xq91cyaN8bdioyib4KfU0180wgltJURI
"""

pip install opencv-python numpy

from google.colab import drive
drive.mount('/content/drive/')

import tensorflow

import tensorflow
print(tensorflow.__version__)

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import cv2

from google.colab.patches import cv2_imshow

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "/content/drive/My Drive/weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = "/content/drive/My Drive/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load Caffe model cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# read the desired image
image = cv2.imread("/content/drive/My Drive/weights/IMG_0014.jpg")
print (image)

#crop_img = image[80:280, 150:330]
#cv2.imshow("cropped", crop_img)
#cv2.waitKey(0)

plt.imshow(image)
plt.show()

# get width and height of the image
h, w = image.shape[:2]
# gaussian blur kernel size depends on width and height of original image
kernel_width = (w // 7) | 1
kernel_height = (h // 7) | 1
# preprocess the image: resize and performs mean subtraction
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
# set the image into the input of the neural network
model.setInput(blob)
# perform inference and get the result
output = np.squeeze(model.forward())
for i in range(0, output.shape[0]):
    confidence = output[i, 2]
    # get the confidence
    # if confidence is above 40%, then blur the bounding box (face)
    if confidence > 0.4:
        # get the surrounding box cordinates and upscale them to original image
        box = output[i, 3:7] * np.array([w, h, w, h])
        # convert to integers
        start_x, start_y, end_x, end_y = box.astype(np.int)
        # get the face image
        face = image[start_y: end_y, start_x: end_x]
        # apply gaussian blur to this face
        face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
        # put the blurred face into the original image
        image[start_y: end_y, start_x: end_x] = face

plt.imshow(image)