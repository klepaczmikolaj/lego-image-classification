# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import json
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

image_size = 160

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
parser.add_argument("-i", "--images", required=True,
                    help="path to input image path")
args = parser.parse_args()

print("Model: {mod}, image path: {path}".format(mod=args.model, path=args.images))


model = load_model(args.model)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        args.images,
        target_size=(image_size, image_size),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)

y_true = test_generator.classes
y_true = label_binarize(y_true, classes=list(range(20)))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(20):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

chart_count = 4
curves_per_chart = 5
colors = ['r', 'b', 'g', 'c', 'm']

for chart in range(chart_count):
    plt.figure(chart + 1)
    for i in range(curves_per_chart):
        cur = chart * curves_per_chart + i
        plt.plot(fpr[cur], tpr[cur], color=colors[i], lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(cur, roc_auc[cur]))
    plt.legend()
    plt.savefig(args.model + '_' + str(chart) + '.png')


'''
# load image

image = cv2.imread(args.images)

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
'''
