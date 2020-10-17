import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from mlflow_log import MLFlowCallback
from model_architecture import create_model


class ClassficationModel:
    def __init__(self, config, num_classes, n_training_points):
        self.n_training_points = n_training_points
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = num_classes
        self.config = config
        if config["model"]["load_name"] != "None":
            self.model = self._load_combined_model(config["data"]["artifact_dir"], config["model"]["load_name"])
        else:
            self.model = create_model(config, self.num_classes, n_training_points)
        print(self.model.summary())

    def train(self, train_data_generator, val_data_generator):
        if self.config["logging"]["log_experiment"]:
            callbacks = [MLFlowCallback(self.config)]
        else:
            callbacks = []
        steps_per_epoch = int(self.n_training_points / train_data_generator.batch_size)
        self.model.fit(
            train_data_generator,
            epochs=self.config["model"]["epochs"],
            steps_per_epoch=steps_per_epoch,
            callbacks=[callbacks],
            validation_data=val_data_generator
        )

    def test(self, test_data_generator):
        metrics = self.model.evaluate(
            test_data_generator,
            steps=test_data_generator.n / self.batch_size,
            return_dict=True
        )
        return metrics

    def predict(self, test_data_generator, output_dir):
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def _load_combined_model(self, artifact_path: str = "./models/", name: str = "cnn"):
        model_path = os.path.join(artifact_path, "models")
        feature_extractor = tf.keras.models.load_model(os.path.join(model_path, name + "_feature_extractor.h5"))
        head = tf.keras.models.load_model(os.path.join(model_path, name + "_head.h5"))
        model = tf.keras.models.Sequential([feature_extractor, head])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(self.config["model"]["learning_rate"]),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        model.summary()
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
