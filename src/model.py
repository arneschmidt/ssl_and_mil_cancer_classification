import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from mlflow_log import MLFlowCallback, format_metrics_for_mlflow
from model_architecture import create_model
from sklearn.utils import class_weight
from utils.mil_utils import combine_pseudo_labels_with_instance_labels, get_data_generator_with_targets, \
    get_data_generator_without_targets, get_one_hot_training_targets
from utils.save_utils import save_dataframe_with_output, save_metrics_artifacts
from metrics import MetricCalculator
from typing import Dict, Optional, Tuple
from data import DataGenerator


class Model:
    """
    Class that contains the classification model. Wraps keras model.
    """
    def __init__(self, config: Dict, n_training_points: int):
        self.n_training_points = n_training_points
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = config["data"]["num_classes"]
        self.config = config
        self.model = create_model(config, self.num_classes, n_training_points)
        if config["model"]["load_model"] != 'None':
            self._load_combined_model(config["model"]["load_model"])
        self._compile_model()

        print(self.model.layers[0].summary())
        print(self.model.layers[1].summary())

    def train(self, data_gen: DataGenerator):
        """
        Train the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        """
        # initialize callback for training procedure: logging and metrics calculation at the end of each epoch
        metric_calculator = MetricCalculator(self.model, data_gen, self.config, mode='val')
        mlflow_callback = MLFlowCallback(self.config, metric_calculator)
        callbacks = [mlflow_callback]

        # initialize generators with weak and strong augmentation
        train_generator_weak_aug = data_gen.train_generator_weak_aug
        train_generator_strong_aug = data_gen.train_generator_strong_aug
        class_weights = None

        steps_all = np.ceil(train_generator_strong_aug.n / self.batch_size)
        steps_positive_bags_only = np.ceil(train_generator_weak_aug.n / self.batch_size)
        predictions_indices = np.array(data_gen.train_df_weak_aug['index'])

        num_pseudo_labels = self.config['data']['positive_pseudo_instance_labels_per_bag']
        label_weights = self.config['data']['label_weights']

        # loop for training per epochs
        for epoch in range(self.config["model"]["epochs"]):
            # Semi-supervised MIL case: make predictions, produce pseudo labels and soft labels
            if self.config['data']['supervision'] == 'mil':
                print('Make predictions to produce pseudo labels..')
                predictions = self.model.predict(train_generator_weak_aug, batch_size=self.batch_size, steps=steps_positive_bags_only, verbose=1)
                training_targets, sample_weights = combine_pseudo_labels_with_instance_labels(predictions, predictions_indices,
                                                                                              data_gen.train_df, num_pseudo_labels, label_weights)
            # Supervised case: get one hot targets from labels
            else:
                training_targets, sample_weights = get_one_hot_training_targets(data_gen.train_df, label_weights,
                                                                                self.num_classes)
            # Optional: class-weighting based on groundtruth and estimated labels
            if self.config["model"]["class_weighted_loss"]:
                class_weights = self._calculate_class_weights(training_targets)

            # Use gt, pseudo and soft labels to train based on the strongly augmented images
            train_mil_generator = get_data_generator_with_targets(train_generator_strong_aug, training_targets, sample_weights)
            self.model.fit(
                train_mil_generator,
                epochs=epoch+1,
                class_weight=class_weights,
                initial_epoch=epoch,
                steps_per_epoch=steps_all,
                callbacks=[callbacks],
            )

    def test(self, data_gen: DataGenerator):
        """
        Test the model with the parameters specified in the config. Log progress to mlflow (see README)
        :param data_gen: data generator object to provide the image data generators and dataframes
        :return: dict of metrics from testing
        """
        metric_calculator = MetricCalculator(self.model, data_gen, self.config, mode='test')
        metrics, artifacts = metric_calculator.calc_metrics()
        save_metrics_artifacts(artifacts, self.config['output_dir'])
        return metrics

    def predict(self, data_gen: DataGenerator):
        """
        Save predictions of a single random batch for demonstration purposes.
        :param data_gen:  data generator object to provide the image data generators and dataframes
        """
        image_batch = data_gen.test_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, self.config['output_dir'])

    def predict_features(self, data_gen: DataGenerator):
        """
        Save feature output of model for potential bag-level classifier. Currently not used.
        :param data_gen:  data generator object to provide the image data generators and dataframes
        """
        output_dir = self.config['output_dir']
        train_gen = data_gen.train_generator_strong_aug
        train_steps = np.ceil(train_gen.n / train_gen.batch_size)
        feature_extractor = self.model.layers[0]
        train_features = feature_extractor.predict(train_gen, steps=train_steps)
        train_predictions = self.model.predict(train_gen, steps=train_steps)
        save_dataframe_with_output(data_gen.train_df, train_predictions, train_features, output_dir, 'Train_features')

        val_steps = np.ceil(data_gen.validation_generator.n / data_gen.validation_generator.batch_size)
        val_gen_images = get_data_generator_without_targets(data_gen.validation_generator)
        val_features = feature_extractor.predict(val_gen_images, steps=val_steps)
        val_predictions = self.model.predict(val_gen_images, steps=val_steps)
        save_dataframe_with_output(data_gen.val_df, val_predictions, val_features, output_dir, 'Test_features')

    def _calculate_class_weights(self, training_targets: np.array):
        """
        Calculate class weights based on gt, pseudo and soft labels.
        :param training_targets: gt, pseudo and soft labels (fused)
        :return: class weight dict
        """
        class_predictions = np.argmax(training_targets, axis=1)
        classes = np.arange(0,self.num_classes)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=class_predictions)
        class_weights = {}
        for class_id in classes:
            class_weights[class_id] = class_weights_array[class_id]
        return class_weights

    def _compile_model(self):
        """
        Compile keras model.
        """
        input_shape = (self.batch_size, self.config["data"]["image_target_size"][0],
                       self.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)

        if self.config['model']['optimizer'] == 'sgd':
            optimizer = tf.optimizers.SGD(learning_rate=self.config["model"]["learning_rate"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=self.config["model"]["learning_rate"])

        if self.config['model']['loss_function'] == 'focal_loss':
            loss = tfa.losses.SigmoidFocalCrossEntropy()
        else:
            loss = 'categorical_crossentropy'


        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    # tfa.metrics.F1Score(num_classes=self.num_classes),
                                    # tfa.metrics.CohenKappa(num_classes=self.num_classes, weightage='quadratic')
                                    ])

    def _load_combined_model(self, artifact_path: str = "./models/"):
        model_path = os.path.join(artifact_path, "models")
        self.model.layers[0].load_weights(os.path.join(model_path, "feature_extractor.h5"))
        self.model.layers[1].load_weights(os.path.join(model_path, "head.h5"))
        self.model.summary()

    def _save_predictions(self, image_batch: np.array, predictions: np.array, output_dir: str):
        for i in range(image_batch[0].shape[0]):
            plt.figure()
            image = image_batch[0][i]
            ground_truth = image_batch[1][i][1]
            prediction = predictions[i][1]
            plt.imshow(image.astype(int))
            plt.title("Ground Truth: " + str(ground_truth) + "    Prediction: " + str(prediction))
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, str(i) + ".png"))

