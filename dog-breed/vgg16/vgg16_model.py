import os
import csv
import numpy as np
import pandas as pd

#pretrained model imports
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# WORKS, JUST MOVING TO SPEED UP BUILD TEMP
# ----------------------------------------

# # TRAIN
# # print('loading training data')
# train_datagen = ImageDataGenerator(rescale=1./255,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip = True)
# training_set = train_datagen.flow_from_directory('../dataset/train',
# target_size=(im_size, im_size),
# batch_size=32,
# class_mode = 'categorical')

# # TEST
# # print('loading testing data')
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_set = test_datagen.flow_from_directory('../dataset/test',
# target_size=(90, 90),
# batch_size=32,
# class_mode = 'input')

#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
 
# #Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')
 
# #Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')
 
# #Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')