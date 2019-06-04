import matplotlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import os
import json
from models import MyModel

matplotlib.use("Agg")

train_data_dir = 'input_data_trim'
output_model_file = 'basic_model.h5'
mapping = 'mapping.json'
validation_split = 0.25
batch_size = 16
epoch_number = 10
image_height = 150
image_width = 150


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
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training') # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    with open(mapping, 'w') as outfile:
        json.dump(validation_generator.class_indices, outfile, sort_keys=True, indent=4)

    return train_generator, validation_generator


def train_model(model_cl):
    (train_generator, validation_generator) = get_generators()
    # train model
    model = model_cl.get_simple_model()
    model.summary()

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epoch_number,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

    print("Model saved to file: {}".format(output_model_file))
    model.save(output_model_file)


if __name__ == "__main__":
    if not os.path.exists(train_data_dir):
        print("Train data directory does not exist, exiting")
        exit(1)
    model_class = MyModel(height=image_height, width=image_width)
    train_model(model_class)
