import matplotlib

matplotlib.use("Agg")

# from imutils import paths
# import random
import shutil
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import img_to_array
# from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import cv2
import os

train_data_dir = 'training_data'
validation_data_dir = 'validation_data'

if not os.path.exists(train_data_dir):
    print("Train data directory does not exist")
    exit(1)
if not os.path.exists(validation_data_dir):


    shutil.copytree(cropped_image_dir, train_data_dir)



# imagePaths = sorted(list(paths.list_images(data_dir)))
# random.seed(123)
# random.shuffle(imagePaths)
# print(imagePaths)
