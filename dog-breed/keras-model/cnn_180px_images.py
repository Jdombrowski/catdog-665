import os
import csv
import numpy as np
import pandas as pd

from keras import optimizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#resizing pixel size, change to check the accuracy of the system
im_size = 180
filter_1_size = 32
kernel_1_size = 3

# TRAIN
# print('loading training data')
train_datagen = ImageDataGenerator(rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../dataset/train',
target_size=(im_size, im_size),
batch_size=32,
class_mode = 'categorical')

# TEST
# print('loading testing data')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('../dataset/test',
target_size=(im_size, im_size),
batch_size=32,
class_mode = 'input')

# ~ model time
model = Sequential()

# ~ CONVOLUTION
model.add(Conv2D(filter_1_size, (kernel_1_size, kernel_1_size), 
input_shape=(im_size, im_size, 3),
activation = 'relu'))
# ~ POOLING 
model.add(MaxPooling2D(pool_size = (2, 2)))

# ~ CONVOLUTION
model.add(Conv2D(filter_1_size, (kernel_1_size, kernel_1_size), 
input_shape=(10, 10, 9),
activation = 'relu'))
# ~ POOLING 
model.add(MaxPooling2D(pool_size = (2, 2)))

# ~ FLATTEN
model.add(Flatten())

# ~ FULL CONNECTION
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 120, activation = 'softmax'))

# ~ COMPILATION
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ~ MODEL LOOKING LIKE A SNACK
# ~ FIT THAT SHIT
model.fit_generator(training_set, 
steps_per_epoch = 500, 
epochs = 20,
validation_data = test_set,
validation_steps = 100
)

