import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.callbacks import ModelCheckpoint
import mlflow


class ClassficationModel:
    def __init__(self, config, num_classes):
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = num_classes
        if config["model"]["load_name"] != "None":
            self._load_combined_model(config["data"]["artifact_dir"], config["model"]["load_name"])
        else:
            self._create_model(config["model"]["architecture"],
                               config["data"]["image_target_size"],
                               self.num_classes)
        print(self.model.summary())

    def train(self, train_data_generator, val_data_generator, save_model_path):
        mlflow.tensorflow.autolog()
        # os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        # checkpoint_callback = ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=1)
        steps_per_epoch = int(train_data_generator.n / train_data_generator.batch_size)
        self.model.fit(
            train_data_generator,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            # callbacks=[checkpoint_callback],
            validation_data=val_data_generator
        )

    def test(self, test_data_generator):
        self.model.evaluate(
            test_data_generator,
            steps=test_data_generator.n/self.batch_size,
        )

    def predict(self, test_data_generator, output_dir):
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def save(self, path: str = "./models/",  name: str = "cnn"):
        self.feature_extractor.save(os.path.join(path, name + "_feature_extractor.h5"))
        self.head.save(os.path.join(path, name + "_head.h5"))

    def _load_combined_model(self, artifact_path: str = "./models/", name: str = "cnn"):
        model_path = os.path.join(artifact_path, "models")
        self.feature_extractor = tf.keras.models.load_model(os.path.join(model_path, name + "_feature_extractor.h5"))
        self.head = tf.keras.models.load_model(os.path.join(model_path, name + "_head.h5"))
        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001), # TODO move to config
            metrics=['accuracy'],
        )
        self.model.summary()

    def _create_model(self, model_architecture, image_size, num_classes):
        input_shape = (image_size[0], image_size[1], 3)

        feature_extractor = Sequential()

        if model_architecture == "mobilenetv2":
            feature_extractor.add(MobileNetV2(include_top=False, input_shape=input_shape, weights=None, pooling='avg'))
        elif model_architecture == "efficientnetb0":
            feature_extractor.add(EfficientNetB0(include_top=False, input_shape=input_shape, weights=None, pooling='avg'))
        elif model_architecture == "simple_cnn":
            feature_extractor.add(SeparableConv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
            for i in range(3):
                feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
                feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
                feature_extractor.add(MaxPool2D(pool_size=(2, 2)))
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(GlobalAveragePooling2D())
        else:
            raise Exception("Choose valid model architecture!")

        head = Sequential()
        head.add(Dense(16, activation="relu"))
        head.add(Dense(int(num_classes), activation="softmax"))

        model = Sequential([feature_extractor, head])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.feature_extractor = feature_extractor
        self.head = head
        self.model = model

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
