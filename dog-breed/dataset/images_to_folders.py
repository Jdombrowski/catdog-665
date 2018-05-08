import pandas as pd
import os
import numpy as np
import shutil

# source is the current directory
# Open dataset file
dataset = pd.read_csv('../labels.csv')
file_names = list(dataset['id'].values)
img_labels = list(dataset['breed'].values)

folders_to_be_created = np.unique(img_labels)
# print(folders_to_be_created)

source = os.getcwd()

for new_path in folders_to_be_created:
    if not os.path.exists(new_path):
        os.makedirs(new_path)

folders = folders_to_be_created.copy()

for f in range(0, len(file_names)):

  current_img = file_names[f]
  current_label = img_labels[f]
  
  if (os.path.isfile(current_img+".jpg")):
    shutil.move((str(current_img)+".jpg"), str(current_label) + "/")


  
  