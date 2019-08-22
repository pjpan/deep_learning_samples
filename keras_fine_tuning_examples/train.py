#!/usr/bin/env python
# encoding: utf-8
'''
@author: ppj
@contact: immortalness@gmail.com
@file: train.py
@time: 2019-08-19 01:13
@desc:
'''
import matplotlib
matplotlib.use("Agg")

# import packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import *
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
from utils import config
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# plotting traning history
def plot_training(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


# devive the paths to the training, valicatioin, and testing
trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.TEST])

# determine the total number of image paths in training
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

# initialize the training data augmenttation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator()


# define the Imagenet mean subtraction and set the mean subtraction value
# for each of the data augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


# initialize the training generator
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    valPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BATCH_SIZE)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BATCH_SIZE)

# load the vgg16 network ,ensuring head fc layer sets

baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of
# the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# place the head fc model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them
for layer in baseModel.layers:
    layer.trainable = False

# compile our models
print("[INFO] compiling model ... ")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network for a few epochs
print("[INFO] training head ...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain//config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal//config.BATCH_SIZE,
    epochs=5
)

# reset the testing generator and evaluate the network
print("[INFO] evaluating after fine-tuning network head...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
        steps=(totalTest// config.BATCH_SIZE)+1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))


# plot the traning history via plot_training
plot_training(H, 5, config.WARMUP_PLOT_PATH)

# let's process to unfreeze the finnal set of CONV layrs in the base model layers
# reset our data generators
trainGen.reset()
valGen.reset()

# now that the head FC layers have been trained/initialized,unfreeze the final
# set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

# loop over the layers in the model and show which ones are trainalbe
for layer in baseModel.layers:
    print("{}: {}".format(layer, layer.trainable))


# for the changes to the model to take affect we need to recompile
# the model, this time using SGB with a very small learning rate
print("[INFO] re-compiling modle... ")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the model again
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // config.BATCH_SIZE,
    epoch=5)

# reset the tesing generator and then use our trained model
# make predictions on the data
print("[INFO] evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
                                steps=(totalTest // config.BATCH_SIZE)+1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
                         target_names=testGen.class_indices.keys()))
plot_training(H, 5, config.UNFROZEN_PLOT_PATH)

# serialize the model to disk
print("[INFO] serializing network...")
model.save(config.MODEL_PATH)

 # to see the network structure
print(baseModel.summary())
print(headModel.summary())
print(model.summary())