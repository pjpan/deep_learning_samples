#!/usr/bin/env python
# encoding: utf-8
'''
@author: ppj
@contact: immortalness@gmail.com
@file: config.py
@time: 1/30/20 3:13 PM
@desc:
'''

import os

# 设置模型文件的正负样本的文件路径
FIRE_PATH = os.path.sep.join(["Robbery_Accident_Fire_Database2", "Fire"])
NON_FIRE_PATH = "spatial_envelope_256x256_static_8outdoorcategories"

# 2分类，没有和有火
CLASSES = ["Non_Fire", "fire"]

# define the size  of the training and test sets
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

# define the initial learning rate
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

# 设置模型存储路径和训练结果的存储路径
MODEL_PATH = os.path.sep.join(["output", "fire_dection.model"])
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "traing_plot.png"])

# 设置预测的文件目录和抽样的预测个数
# output/examples文件需要提前创建
OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50