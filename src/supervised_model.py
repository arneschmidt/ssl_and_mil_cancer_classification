import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
from mlflow_log import MLFlowCallback
from model_architecture import create_model
from sklearn.utils import class_weight
from metrics import MetricCalculator


class SupervisedModel:
    def __init__(self, config, n_training_points):
        self.n_training_points = n_training_points
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = config["data"]["num_classes"]
        self.config = config
        self.model = create_model(config, self.num_classes, n_training_points)
        if config["model"]["load_model"]:
            self._load_combined_model(config["output_dir"])
        self._compile_model()

        print(self.model.summary())

    def train(self, data_gen):
        train_data_generator = data_gen.train_generator
        val_data_generator = data_gen.validation_generator
        metric_calculator = MetricCalculator(self.model, data_gen, self.config, mode='val')
        mlflow_callback = MLFlowCallback(self.config, metric_calculator)
        callbacks = [mlflow_callback]

        if self.config["model"]["class_weighted_loss"] and self.config['model']['use_fixed_class_weights'] is False:
            class_weights_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_data_generator.classes),
                y=train_data_generator.classes)
            class_weights = {}
            for class_id in train_data_generator.class_indices.values():
                class_weights[class_id] = class_weights_array[class_id]
        elif self.config["model"]["class_weighted_loss"] and self.config['model']['use_fixed_class_weights']:
            class_weights = self.config['data']['fixed_class_weights']
        else:
            class_weights = None

        steps_per_epoch = int(self.n_training_points / train_data_generator.batch_size)
        self.model.fit(
            train_data_generator,
            epochs=self.config["model"]["epochs"],
            class_weight=class_weights,
            steps_per_epoch= steps_per_epoch,
            callbacks=[callbacks],
            validation_data=val_data_generator
        )

    def test(self, data_gen):
        test_data_generator = data_gen.test_generator
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

    def predict(self, data_gen):
        test_data_generator = data_gen.test_generator
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, self.config['output_dir'])

    def _compile_model(self):
        input_shape = (self.batch_size, self.config["data"]["image_target_size"][0],
                       self.config["data"]["image_target_size"][1], 3)
        self.model.build(input_shape)
        self.model.compile(optimizer='adam',
                           loss=['categorical_crossentropy'],
                           metrics=['accuracy',
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(),
                                    tfa.metrics.F1Score(num_classes=self.num_classes),
                                    tfa.metrics.CohenKappa(num_classes=self.num_classes)])

    def _load_combined_model(self, artifact_path: str = "./models/"):
        model_path = os.path.join(artifact_path, "models")
        self.model.layers[0].load_weights(os.path.join(model_path, "feature_extractor.h5"))
        self.model.layers[1].load_weights(os.path.join(model_path, "head.h5"))
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
