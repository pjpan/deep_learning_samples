#!/usr/bin/env python
# encoding: utf-8
'''
@author: ppj
@contact: immortalness@gmail.com
@file: build_dataset.py
@time: 2019-08-17 10:22
@desc:
'''
# import packages
from utils import config
from imutils import paths
import shutil
import os

# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
    imagePaths = list(paths.list_images(p))

    # loop over the image paths
    for imagePaths in imagePaths:
        # extract class label from the filename
        filename = imagePaths.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])]

        # construct the path to the output directory
        dirPath = os.path.sep.join([config.BASE_PATH, split, label])

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # construct the path to the output image file and copy it
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePaths, p)




