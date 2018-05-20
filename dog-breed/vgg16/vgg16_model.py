import os
import sys
import csv
import numpy as np
import pandas as pd

from keras.callbacks import CSVLogger
from keras import optimizers
from keras.applications import vgg16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, model_from_json, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

'''
DATA SETUP
'''

number_of_dense_layers = int(sys.argv[1]) or 1
img_size = int(sys.argv[2]) or 256
shear_range = float(sys.argv[3]) or 0.0
zoom_range = float(sys.argv[4]) or 0.0
batch_size = 73

# notes: the actual number of data size
train_samples_size = 10222
test_samples_size = 10358

epochs = 140
steps = train_samples_size // batch_size
validation_steps = test_samples_size // batch_size

# Directories
training_dir = '../dataset/train'
testing_dir = '../dataset/test'

# TRAINING DATA
train_datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=shear_range,
  zoom_range=zoom_range,
  horizontal_flip = True)

training_set = train_datagen.flow_from_directory(
  training_dir,
  target_size=(img_size, img_size),
  batch_size=batch_size,
  class_mode = 'categorical')

# TESTING DATA
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
  testing_dir,
  target_size=(img_size, img_size),
  batch_size=batch_size,
  class_mode = 'input')

'''
BUILD and FINE-TUNE THE MODEL
'''
#Load the VGG16 model
# - set include_top=False to not include the 3 fully-connected layers at the top of the network
# - reshape the input size to 90x90
vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_size,img_size,3))

# Create a new top model
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))

for i in range(0, number_of_dense_layers):
  top_model.add(Dense(units = 120, activation = 'relu'))

top_model.add(Dropout(0.5))
top_model.add(Dense(120, activation='softmax'))

# Transform VGG16 model into a Sequential model by adding its existing layers
model = Sequential()
for layer in (vgg16_model.layers):
  model.add(layer)

# Freeze the existing layers to prevent further training
for layer in model.layers:
  layer.trainable = False

# Add the new top model
model.add(top_model)

# top_model.summary()
csv_logger = CSVLogger("training.log", append="True")

'''
COMPILE and FIT
'''
# Compile the model
model.compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = ['accuracy'])

# Fit the model
model.fit_generator(
  training_set,
  steps_per_epoch = steps,
  epochs = epochs,
  validation_data = test_set,
  validation_steps = validation_steps,
  callbacks=[csv_logger])