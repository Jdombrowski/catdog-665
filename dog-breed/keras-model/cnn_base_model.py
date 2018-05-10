import csv
import os
import sys

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator	

# TODO: Track how long it takes this to run in addition to saving its results

class Network:

  # All networks have the same number of training images

  NUM_TRAINING_IMAGES = 10220

  # All of our images are in three color channels

  image_channels = 3

  # 3x3 convolutional matrix to process each individual pixel
  # of an input image (and/or node?)

  kernel_size = 3

  # Constructor
  
  def __init__(self):

    # Names are used when saving and loading networks
    self.name = "NONE"

    # By default, we give a network an empty sequential model 
    # as opposed to from an API
    self.model = Sequential()

    # Images are resized to 90x90 for uniformity and efficiency
    self.image_size = 90

    # Batch size is the number of training examples in one epoch.
    # It is one parameter that is generally determined by trial and
    # error, and should be equal to the total amount of training
    # data divided by the number of epochs. 

    # To start, we will try two batch sizes, 73 and 140, which are 
    # the least different factors of 10220 (our total number of 
    # training images. For each of these, the number of epochs will 
    # be equal to the other value.

    self.batch_size = 73

    # And now we can generate the data

    self.training_data = self.generate_training_data()

    self.testing_data = self.generate_testing_data()

    # TODO: Number of output filters by layer?

  # TODO: Tried to write a function which would allow us to call print(Network). It didn't work

  # Saves the network to two files
  
  def save(self):
    while self.name == "NONE":
      self.name = input("Please enter network name: ")
    model_json = self.model.to_json()
    json_name = self.name + ".json"
    weights_name = self.name + ".h5"
    try:	
      with open(json_name, "w") as json_file:
        json_file.write(model_json)
        json_file.close()
      # serialize weights to HDF5
      self.model.save_weights(weights_name)
      print("Saved model to disk")
    except:
      print("Unexpected error:", sys.exc_info()[0])

  # Saves the network to two files with specified names

  def save_as(self, file_name):
    while "." in file_name:
      print("To save the model/weights with specific filenames, please enter the name of the",
        " files without their extensions (i.e. newModel rather than newModel.json)")
      return
    model_json = self.model.to_json()
    json_name = self.name + ".json"
    weights_name = self.name + ".h5"
    try:	
      with open(json_name, "w") as json_file:
        json_file.write(model_json)
        json_file.close()
      # serialize weights to HDF5
      self.model.save_weights(weights_name)
      print("Saved model to disk")
    except:
      print("Unexpected error:", sys.exc_info()[0])

  # Loads the network from a file 

  # TODO: Does not set the network name. Should it?
  
  def load_from_file(self, file):
    try:
      with open(file, 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        weights_file = file + ".h5"
        loaded_model.load_weights(weights_file)
        self.model = loaded_model
      print("Loaded model from file " + file)
    except:
      print("Unexpected error: ", sys.exc_info()[0], " model not initialized")

  # TODO: Load weights from another weight file?

  # Generates training data, called in constructor

  def generate_training_data(self):

    # TODO: This should pull its arguments from a config file

    training_data_generator = ImageDataGenerator(rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip = True)

    return training_data_generator.flow_from_directory('../dataset/train',
      target_size = (self.image_size, self.image_size),
      batch_size = self.batch_size,
      class_mode = 'categorical',
      shuffle=True)

  # Generates testing data, called in constructor

  def generate_testing_data(self):

    # TODO: This should pull its arguments from a config file

    test_data_generator = test_datagen = ImageDataGenerator(rescale=1./255)

    return test_data_generator.flow_from_directory('../dataset/test',
      target_size = (self.image_size, self.image_size),
      batch_size = self.batch_size,
      class_mode = 'input',
      shuffle=True)

  # Adds layers: 
  def add(self, layer):
    self.model.add(layer)

  # Adds a convolutional layer with 32 output nodes, a kernel according to the class kernel size, 
  # a relu activation function, and whatever other parameters we pass in

  def addConvLayer(self, **kwargs):
    self.model.add(Conv2D(32, (Network.kernel_size, Network.kernel_size), activation = 'relu', **kwargs))
    
  # Compiles the model with the given arguments

  def compile(self, **kwargs):
    self.model.compile(**kwargs)

  def fit_generator(self, log_file_name, **kwargs):
    # Check that the batch size and number of epochs were set to maximize the training data
    assert (kwargs.get("epochs") <= Network.NUM_TRAINING_IMAGES/self.batch_size), "Batch size must be NUM_TRAINING_IMAGES/epochs" 
    # Generate the fit according to our training data, testing data, and other params
    # Epochs must be passed in to this function when it is called

  #define log file
    csv_logger = CSVLogger(log_file_name, append="True")
    self.model.fit_generator(self.training_data, validation_data = self.testing_data,callbacks=[csv_logger] ,**kwargs)




# END CLASS DECLARATION

# CREATES, TRAINS, AND TESTS NEW NETWORK

nn = Network() # Has the training/testing data and a sequential model

# Have to add the layers 
# TODO: Should be able to pull them from a list or something? 

# TODO: Create an "add input layer" command?

nn.addConvLayer(input_shape=(90, 90, 3))

nn.add(MaxPooling2D(pool_size = (4, 4)))

nn.addConvLayer()

nn.add(MaxPooling2D(pool_size = (2, 2)))

nn.addConvLayer()

nn.add(MaxPooling2D(pool_size = (2, 2)))

nn.add(Flatten())

for i in range(0, 1):
  nn.add(Dense(units = 120, activation = 'relu'))

nn.add(Dense(units = 120, activation = 'softmax'))

nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

nn.model.summary()

# Steps per epoch default `None` is equal to the number of samples in your dataset 
# divided by the batch size, so I deleted what we were using to allow it be that default
nn.fit_generator('training.log', epochs = 10)
