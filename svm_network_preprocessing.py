from sklearn.svm import SVC
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

train_data_dir = 'input_data_trim'
test_data_dir = 'input_data_test'

mapping = 'mapping.json'
validation_split = 0.2
batch_size = 16
epoch_number = 5
image_size = 160

model = load_model('all_trim_model_trim.h5')

model_feat = Model(inputs=model.input,outputs=model.get_layer('global_average_pooling2d_2').output)

test_datagen = ImageDataGenerator(rescale=1./255)
train_batches = test_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_size, image_size),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

test_batches = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_size, image_size),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = train_batches.filenames
nsteps = len(filenames)
feat_train = model_feat.predict_generator(train_batches, steps = nsteps)
print(feat_train.shape)

filenames = test_batches.filenames
nsteps = len(filenames)
feat_test = model_feat.predict_generator(test_batches, steps = nsteps)
print(feat_test.shape)

np.save('feat_train.npy', feat_train)
np.save('feat_test.npy', feat_test)
np.save('train_batches.npy', train_batches)
np.save('test_batches.npy', test_batches)