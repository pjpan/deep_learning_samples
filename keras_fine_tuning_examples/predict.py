#!/usr/bin/env python
# encoding: utf-8
'''
@author: ppj
@contact: immortalness@gmail.com
@file: predict.py
@time: 2019-08-21 22:32
@desc: make predictions with fine-tuning and keras
'''

# import the necessary packages
from keras.models import load_model
from utils import config
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                required=True, help="image path ")
args = vars(ap.parse_args())

# laod the input image and then clone it so we can draw on it later
image = cv2.imread(args["image"])
output = image.copy()
output = imutils.resize(output, width=400)
# our model was trained on RGB orderd images but opencv
# represents images in BGR, so swap the channels, and resize to 224*224
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))

# convert the image to a floating point data type and perform mean subtraction
# image data process
image = image.astype("float32")
mean = np.array([123.6, 116.779, 103.939][::1], dtype="float32")
image-=mean

# Note: When we perform inference using a custom prediction script, if the results are unsatisfactory nine times out of ten it is due to improper preprocessing. Typically having color channels in the wrong order or forgetting to perform mean subtraction altogether will lead to unfavorable results. Keep this in mind when writing your own scripts.

# load the trained model from disk
# perform inference
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# pass the image through the network to obtain our predictions
preds = model.predict(np.expend_dims(image, axis=0))[0]
i = np.argmax(preds)
label = config.CLASSES[i]

# draw the prediction on the output image
text = "{}: {:.2f%}".format(label, preds[i] * 100)
cv2.putText(output, text, (3,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0,255,0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)



