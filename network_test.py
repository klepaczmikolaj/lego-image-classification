# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import json

image_height = 150
image_width = 150

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
parser.add_argument("-i", "--image", required=True,
                    help="path to input image")
args = parser.parse_args()

print("Model: {mod}, image path: {path}".format(mod=args.model, path=args.image))

# load image
image = cv2.imread(args.image)

# pre-process the image for classification
image = cv2.resize(image, (image_height, image_width))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# classify image
model = load_model(args.model)
result = model.predict(image)
res_class = result.argmax(axis=-1)[0]

with open("mapping.json") as f_in:
    mapping = json.loads(f_in.read())
for name, no in mapping.items():
    if no == res_class:
        print(name)
