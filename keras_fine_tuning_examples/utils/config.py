#!/usr/bin/env python
# encoding: utf-8
'''
@author: ppj
@contact: immortalness@gmail.com
@file: config.py
@time: 2019-08-17 10:10
@desc:
'''

import os

# original input directory of images
ORIG_INPUT_DATASET = "Food-5K"

BASE_PATH = "dataset"

# define trainning、val、eval
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# set batch size and fine-tuning
BATCH_SIZE =32

#set model file
MODEL_PATH = os.path.sep.join(["output", "food5k.model"])

# set the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup.png"])

CLASSES = ["是", "否"]