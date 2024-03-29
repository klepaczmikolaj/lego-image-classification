from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import matplotlib.pyplot as plt
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

