from keras import applications, optimizers
from keras.models import Model
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D


class MyModel:
    def __init__(self, size):
        self.size = size

    def get_simple_model(self):
        (input_tensor, base_model) = self.get_base_model()

        for layer in base_model.layers:
            layer.trainable = False

        output_tensor = Dense(20, activation='softmax')(base_model.output)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def get_conv_learn_model(self):
        (input_tensor, base_model) = self.get_base_model()

        for layer in base_model.layers[:-4]:
            layer.trainable = False

        output_tensor = Dense(20, activation='softmax')(base_model.output)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def get_all_net_trim_model(self):
        (input_tensor, base_model) = self.get_base_model()

        for layer in base_model.layers:
            layer.trainable = True

        trim = base_model.layers[-6].output
        output = GlobalAveragePooling2D()(trim)
        output_tensor = Dense(20, activation='softmax')(output)
        # UNTRIM
        # output_tensor = Dense(20, activation='softmax')(base_model.output)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def get_svm_model(self):
        (input_tensor, base_model) = self.get_base_model()

        for layer in base_model.layers:
            layer.trainable = False

        output_tensor = Dense(20, activation='softmax')(base_model.output)
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def get_base_model(self):
        input_tensor = Input(shape=(self.size, self.size, 3))
        base_model = applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=(self.size, self.size, 3),
            pooling='avg')
        return input_tensor, base_model