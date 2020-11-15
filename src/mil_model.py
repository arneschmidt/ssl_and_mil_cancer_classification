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
from mlflow_log import MLFlowCallback, format_metrics_for_mlflow
from model_architecture import create_model
from sklearn.utils import class_weight
from utils.mil_utils import combine_pseudo_labels_with_instance_labels, get_data_generator_with_targets, \
    get_data_generator_without_targets
from utils.save_utils import save_dataframe_with_output

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

        print(self.model.layers[0].summary())
        print(self.model.layers[1].summary())

    def train(self, data_gen):
        if self.config["logging"]["log_experiment"]:
            callbacks = [MLFlowCallback(self.config)]
        else:
            callbacks = []

        if self.config["model"]["class_weighted_loss"]:
            class_weights = self.config['data']['mil_class_weights']
        else:
            class_weights = None

        train_generator_weak_aug = data_gen.train_generator_weak_aug
        train_generator_strong_aug = data_gen.train_generator_strong_aug
        steps = np.ceil(self.n_training_points / train_generator_weak_aug.batch_size)

        for epoch in range(self.config["model"]["epochs"]):
            print('Make predictions to produce pseudo labels..')
            predictions = self.model.predict(train_generator_weak_aug, batch_size=self.batch_size, steps=steps)
            training_targets = combine_pseudo_labels_with_instance_labels(predictions, data_gen,
                                                                          self.config['data']['positive_pseudo_instance_labels_per_bag'])

            train_mil_generator = get_data_generator_with_targets(train_generator_strong_aug, training_targets)
            self.model.fit(
                train_mil_generator,
                epochs=epoch+1,
                class_weight=class_weights,
                initial_epoch=epoch,
                steps_per_epoch=steps,
                callbacks=[callbacks],
                validation_data=data_gen.validation_generator
            )

    def test(self, data_gen):
        metrics = self.model.evaluate(
            data_gen.test_generator,
            steps=data_gen.test_generator.n / self.batch_size,
            return_dict=True
        )
        metrics = format_metrics_for_mlflow(metrics)
        return metrics

    def predict(self, data_gen, output_dir):
        image_batch = data_gen.test_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def predict_features(self, data_gen, output_dir):
        train_steps = np.ceil(self.n_training_points / data_gen.train_generator_weak_aug.batch_size)
        feature_extractor = self.model.layers[0]
        train_features = feature_extractor.predict(data_gen.train_generator_weak_aug, steps=train_steps)
        train_predictions = self.model.predict(data_gen.train_generator_weak_aug, steps=train_steps)
        save_dataframe_with_output(data_gen.train_df, train_predictions, train_features, output_dir, 'Train_features')

        val_steps = np.ceil(data_gen.validation_generator.n / data_gen.validation_generator.batch_size)
        val_gen_images = get_data_generator_without_targets(data_gen.validation_generator)
        val_features = feature_extractor.predict(val_gen_images, steps=val_steps)
        val_predictions = self.model.predict(val_gen_images, steps=val_steps)
        save_dataframe_with_output(data_gen.val_df, val_predictions, val_features, output_dir, 'Test_features')

    def _compile_model(self):
        input_shape = (self.batch_size, self.config["data"]["image_target_size"][0],
                       self.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)

        if self.config['model']['optimizer'] == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=self.config["model"]["learning_rate"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=self.config["model"]["learning_rate"])
        self.model.compile(optimizer=optimizer,
                           loss=['categorical_crossentropy'],
                           metrics=['accuracy',
                                    # tf.keras.metrics.Precision(),
                                    # tf.keras.metrics.Recall(),
                                    tfa.metrics.F1Score(num_classes=self.num_classes),
                                    tfa.metrics.CohenKappa(num_classes=self.num_classes, weightage='quadratic')])

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

