import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from mlflow_log import MLFlowCallback
from model_architecture import create_model
from sklearn.utils import class_weight
from utils.mil_utils import combine_pseudo_labels_with_instance_labels, get_data_generator_with_targets


class MILModel:
    def __init__(self, config, num_classes, n_training_points):
        self.n_training_points = n_training_points
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = num_classes
        self.config = config
        self.model = create_model(config, self.num_classes, n_training_points)
        if config["model"]["load_name"] != "None":
            self._load_combined_model(config["data"]["artifact_dir"], config["model"]["load_name"])
        self._compile_model()

        print(self.model.summary())

    def train(self, data_gen):
        class_weights = self.config['data']['mil_class_weights']
        for epoch in range(self.config["model"]["epochs"]):
            train_generator_weak_aug = data_gen.train_generator_weak_aug
            steps = np.ceil(self.n_training_points / train_generator_weak_aug.batch_size)
            predictions = self.model.predict(train_generator_weak_aug, batch_size=self.batch_size, steps=steps)
            training_targets = combine_pseudo_labels_with_instance_labels(predictions, data_gen)

            train_generator_strong_aug = data_gen.train_generator_strong_aug
            train_mil_generator = get_data_generator_with_targets(train_generator_strong_aug, training_targets)
            steps_per_epoch = int(self.n_training_points / train_generator_strong_aug.batch_size)
            self.model.fit(
                train_mil_generator,
                epochs=1,
                class_weight=class_weights,
                initial_epoch=epoch,
                steps_per_epoch= steps_per_epoch,
                # callbacks=[callbacks],
                validation_data=data_gen.validation_generator
            )

    def test(self, test_data_generator):
        metrics = self.model.evaluate(
            test_data_generator,
            steps=test_data_generator.n / self.batch_size,
            return_dict=True
        )
        metrics['f1_mean'] = np.mean(metrics['f1_score'])
        for class_id in range(self.num_classes):
            key = 'f1_class_id_' + str(class_id)
            metrics[key] = metrics['f1_score'][class_id]
        metrics.pop('f1_score')
        return metrics

    def predict(self, test_data_generator, output_dir):
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def _compile_model(self):
        input_shape = (self.batch_size, self.config["data"]["image_target_size"][0],
                       self.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)
        self.model.compile(optimizer='adam',
                           loss=['categorical_crossentropy'],
                           metrics=['accuracy'])
                                    # tf.keras.metrics.Precision(),
                                    # tf.keras.metrics.Recall(),
                                    # tfa.metrics.F1Score(num_classes=self.num_classes),
                                    # tfa.metrics.CohenKappa(num_classes=self.num_classes)])

    def _load_combined_model(self, artifact_path: str = "./models/", name: str = "cnn"):
        model_path = os.path.join(artifact_path, "models")
        self.model.layers[0].load_weights(os.path.join(model_path, name + "_feature_extractor.h5"))
        self.model.layers[1].load_weights(os.path.join(model_path, name + "_head.h5"))
        self.model.summary()

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

