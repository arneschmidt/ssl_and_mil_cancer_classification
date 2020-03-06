import os
import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D, SeparableConv2D
from keras.callbacks import ModelCheckpoint
import mlflow.keras


class ClassficationModel:
    def __init__(self, batch_size, load_model_path, model_architecture):
        self.batch_size = batch_size
        if load_model_path is not "None":
            self.model = load_model(load_model_path)
        else:
            self.model = self._initialize_model(model_architecture)
        print(self.model.summary())

    def train(self, train_data_generator, val_data_generator, save_model_path):
        mlflow.keras.autolog()
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        checkpoint_callback = ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=1)
        self.model.fit_generator(
            train_data_generator,
            steps_per_epoch=1000,
            callbacks=[checkpoint_callback],
            validation_data=val_data_generator,
            validation_steps=100,
            epochs=100
        )

    def test(self, test_data_generator):
        self.model.evaluate_generator(
            test_data_generator,
            steps=test_data_generator.n/self.batch_size,
        )

    def predict(self, test_data_generator, output_dir):
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def _initialize_model(self, model_architecture):
        input_shape = (96, 96, 3)
        model = Sequential()
        if model_architecture == "mobilenetv2":
            model.add(MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg'))
        elif model_architecture == "simple_cnn":
            model.add(SeparableConv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            for i in range(3):
                model.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
                model.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
                model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            model.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            model.add(GlobalAveragePooling2D())
        else:
            raise Exception("Choose valid model architecture!")
        model.add(Dense(256, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _save_predictions(self, image_batch, predictions, output_dir):
        for i in range(image_batch[0].shape[0]):
            plt.figure()
            image = image_batch[0][i]
            ground_truth = image_batch[1][i][1]
            prediction = predictions[i][1]
            plt.imshow(image.astype(int))
            plt.title("Ground Truth: " + str(ground_truth) + "    Prediction: " + str(prediction))
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, str(i) + ".png"))
