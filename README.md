# Blur-faces
we’ll briefly discuss what face blurring is and how we can use OpenCV to anonymize faces in images
# Libraries
 Tensorflow.keras, OpenCV,
Image, ImageOps,
numpy 
matplotlib
# Instruction
you have to download the files and store them in your google drive and change the path to new path.
# Run
Pip install unitiled0.ipynb
# Describtion

In order to blur faces shown in images, you need to first detect these faces and their position in the image.

To get started, installing the required dependencies:

pip3 install opencv-python numpy
Open up a new file and import:
import cv2
import numpy as np
As explained in the face detection tutorial, since we need to initialize our deep learning model to detect faces, we need to get the model architecture along with its pre-trained weights, download them, and put them in the weights folder:

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = "weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel 
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# load Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
The below code reads this image, prepares it, and pass it to the neural network:

# read the desired image
image = cv2.imread("father-and-daughter.jpg")
# get width and height of the image

# gaussian blur kernel size depends on width and height of original image
kernel_width = (w // 7) | 1
kernel_height = (h // 7) | 1

