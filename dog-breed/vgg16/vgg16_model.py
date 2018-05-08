import os
import csv
import numpy as np
import pandas as pd

from keras import optimizers
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

'''
DATA SETUP
'''
#resizing parameter 90x90 pixels, change to check the accuracy of the system
img_size = 90
batch_size = 32

# Directories
training_dir = '../dataset/train'
testing_dir = '../dataset/test'

# TRAINING DATA
# print('loading training data')
train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip = True)

training_set = train_datagen.flow_from_directory(training_dir,
target_size=(img_size, img_size),
batch_size=batch_size,
class_mode = 'categorical')

# TESTING DATA
# print('loading testing data')
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(testing_dir,
target_size=(img_size, img_size),
batch_size=batch_size,
class_mode = 'input')

'''
BUILD and FINE-TUNE THE MODEL
'''
#Load the VGG16 model
vgg16_model = vgg16.VGG16(weights='imagenet')
 
# #Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')
 
# #Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')
 
# #Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')

# Transform VGG16 model into a Sequential model by adding its existing layers
model = Sequential()
for layer in vgg16_model.layers:
  model.add(layer)

# Pop the last default Dense layer of VGG16
model.layers.pop()

# Freeze the existing layers to prevent further training
for layer in model.layers:
  layer.trainable = False

# Add the last Dense layer to classify the number of required dog breeds
model.add(Dense(120, activation='softmax'))

'''
TRAIN THE FINE-TUNED MODEL
'''
# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit_generator(training_set, 
steps_per_epoch = 500, 
epochs = 10,
validation_data = test_set,
validation_steps = 4
)

'''
PREDICT USING THE FINE-TUNED MODEL
'''
