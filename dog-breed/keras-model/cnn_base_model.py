import csv
import os

import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#resizing parameter 90x90 pixels, change to check the accuracy of the system
im_size = 90
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
class_mode = 'categorical',
shuffle=True)

# TEST
# print('loading testing data')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('../dataset/test',
target_size=(90, 90),
batch_size=32,
class_mode = 'input',
shuffle=True)

# ~ model time
model = Sequential()

# ~ CONVOLUTION
model.add(Conv2D(filter_1_size, (kernel_1_size, kernel_1_size), 
input_shape=(90, 90, 3),
activation = 'relu'))
# ~ POOLING 
model.add(MaxPooling2D(pool_size = (4, 4)))

# ~ CONVOLUTION
model.add(Conv2D(filter_1_size, (kernel_1_size, kernel_1_size), 
input_shape=(20, 20, 3),
activation = 'relu'))
# ~ POOLING 
model.add(MaxPooling2D(pool_size = (2, 2)))


# ~ CONVOLUTION
model.add(Conv2D(filter_1_size, (kernel_1_size, kernel_1_size), 
input_shape=(9, 9, 3),
activation = 'relu'))
# ~ POOLING 
model.add(MaxPooling2D(pool_size = (2, 2)))

# ~ FLATTEN
model.add(Flatten())

# ~ FULL CONNECTION
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 120, activation = 'softmax'))

# ~ COMPILATION
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ~ FIT MODEL
model.fit_generator(training_set, 
steps_per_epoch = 500, 
epochs = 40,
validation_data = test_set,
validation_steps = 100
)

model_json = model.to_json()
with open("cnn_model_base.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("cnn_model_base.h5")
print("Saved model to disk")
json_file.close()

# reading previous model, clunky for time being

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

