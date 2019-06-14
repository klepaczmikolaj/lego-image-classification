import matplotlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import os
import json
import argparse
from models import MyModel

matplotlib.use("Agg")

train_data_dir = 'input_data_trim'
models_dir = 'models/'
mapping = 'mapping.json'
validation_split = 0.2
batch_size = 16
epoch_number = 50
image_size = 160

def parse_args():
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help='''specify model type: with one of the following:
                                CLS - only classif layer learn,
                                CNV - learning of last conv layer, 
                                TRM - all network train with trim, 
                                SVM - TRM with SVM''')
    args = parser.parse_args()
    if(args.model not in ['CLS', 'CNV', 'TRM', 'SVM']):
        print("Wrong console argument, exiting")
        exit(1)
    return args


# get augmented train and validation data
def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    with open(mapping, 'w') as outfile:
        json.dump(validation_generator.class_indices, outfile, sort_keys=True, indent=4)

    return train_generator, validation_generator


def train_model(model):
    (train_generator, validation_generator) = get_generators()
    # train model
    model.summary()

    model.fit_generator(
        train_generator,
        verbose=2,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epoch_number,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    print("Model saved to file: {}".format(output_model_file))
    model.save(models_dir + output_model_file)


if __name__ == "__main__":
    if not os.path.exists(train_data_dir):
        print("Train data directory does not exist, exiting")
        exit(1)
    args = parse_args()
    models_class = MyModel(size=image_size)
    if args.model == 'CLS':
        output_model_file = 'simple_model.h5'
        model = models_class.get_simple_model()
    elif args.model == 'CNV':
        output_model_file = 'cnv_model.h5'
        model = models_class.get_conv_learn_model()
    elif args.model == 'TRM':
        output_model_file = 'all_untrim_model.h5'
        model = models_class.get_all_net_trim_model()
    elif args.model == 'SVM':
        output_model_file = 'SVM_model.h5'
        model = models_class.get_svm_model()
    else:
        print("Wrong model, use one of following: 'CLS', 'CNV', 'TRM', 'SVM'")
        exit(1)
    train_model(model)
