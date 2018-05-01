# How to Run
Download the data from the kaggle site 

# [Kaggle](https://www.kaggle.com/c/dog-breed-identification/data)

1. Unzip all four files into the `/dataset` folder.

2. Move `images_to_folders.py` into the `/train` directory

3. Run `python3 images_to_folders.py` to move all images into the appropriate folders. Aside: `test/` does not have labels, so running this script in that dir will only result in error. 

## Introduction
### /datasets
This is where you need to do some setup to ge tthe data in the right form to work with frim the above instructions.

### /keras-models
This has been the majority of my work after the data preprocessing. The base model is the file I have been tweaking/ working (twerking) on, the other files in this folders will all end up being creted from this file, with unique changes for testing.

# HIGH SCORE
## Current high score is 0.729
Convolutional netowork with 2 layers of convolution, 90x90 images, and 5 128 dense layers. 