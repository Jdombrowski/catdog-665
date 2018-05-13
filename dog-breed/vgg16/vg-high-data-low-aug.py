import os
import csv
import numpy as np
import pandas as pd

from keras.callbacks import CSVLogger
from keras import optimizers
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

img_size = 128
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
  rescale=1./255)

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

vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_size,img_size,3))

flatten = Flatten()

vgg16_input = vgg16_model.input
vgg16_output = flatten(vgg16_model.output)
# out2 = new_layer2(flatten(vgg16_model.output))

mod_vgg16_model = Model(vgg16_input, vgg16_output)
type(vgg16_model)

# Transform VGG16 model into a Sequential model by adding its existing layers
model = Sequential()
for layer in (mod_vgg16_model.layers):
  model.add(layer)

# Freeze the existing layers to prevent further training
for layer in model.layers:
  layer.trainable = False

# model.layers.pop()

# Add the last Dense layer to classify the number of required dog breeds
model.add(Dense(120, activation='softmax'))
model.summary()
csv_logger = CSVLogger("training.log", append="True")

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

# model.fit_generator(
#   training_set,
#   steps_per_epoch = 10,
#   epochs = epochs,
#   validation_data = test_set,
#   validation_steps = 5)