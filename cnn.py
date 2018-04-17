# example from https://becominghuman.ai/building-an-image-model-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8
import os

import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
# Importing the Keras libraries and packages
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

loadingModelEnabled = False
model = None

# defining the test data
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

def trainModel():
    model = Sequential()
    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units = 128, activation = 'relu'))

#experiment
    model.add(Dropout(0.2))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Part 2 - Fitting the CNN to the images
    
    model.fit_generator(training_set,
    # steps_per_epoch = 8000,
    steps_per_epoch = 20,
    epochs = 5,
    validation_data = test_set,
    # validation_steps = 2000)
    validation_steps = 5)

if loadingModelEnabled:
    if os.path.isfile('model.json'):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk") 
    else:
        trainModel()
else:
    trainModel()

# Part 3 - Making new predictions
test_image = image.load_img('dataset/single_prediction/sushi.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print ('test image is a dog')
else:
    prediction = 'cat'
    print ('test image is a cat')


# #saving and reloading model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
json_file.close()

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# # Part 5 - Making new predictions from loaded model
# test_image = image.load_img('dataset/single_prediction/sushi.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = loaded_model.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
#     print ('test image is a dog')
# else:
#     prediction = 'cat'
#     print ('test image is a cat')
