from keras import applications, optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout
from keras.layers import Activation, Dropout, Flatten, Dense


class MyModel:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_simple_model(self):
        input_tensor = Input(shape=(self.height, self.width, 3))
        base_model = applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(self.height, self.width, 3),
            pooling='avg')

        for layer in base_model.layers:
            layer.trainable = True  # trainable has to be false in order to freeze the layers

        op = Dense(256, activation='relu')(base_model.output)
        op = Dropout(.25)(op)

        op = Dense(256, activation='relu')(base_model.output)
        op = Dropout(.25)(op)

        output_tensor = Dense(20, activation='softmax')(op)

        model = Model(inputs=input_tensor, outputs=output_tensor)

        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        return model
