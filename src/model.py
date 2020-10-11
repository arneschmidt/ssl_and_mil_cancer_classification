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
from src.mlflow_log import MLFlowCallback


class ClassficationModel:
    def __init__(self, config, num_classes, n_training_points):
        self.n_training_points = n_training_points
        self.batch_size = config["model"]["batch_size"]
        self.num_classes = num_classes
        self.config = config
        if config["model"]["load_name"] != "None":
            self._load_combined_model(config["data"]["artifact_dir"], config["model"]["load_name"])
        else:
            self._create_model(config["model"]["feature_extractor"],
                               config["model"]["head"],
                               config["data"]["image_target_size"],
                               self.num_classes)
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
            steps=test_data_generator.n/self.batch_size,
            return_dict=True
        )
        return metrics

    def predict(self, test_data_generator, output_dir):
        image_batch = test_data_generator.next()
        predictions = self.model.predict(image_batch[0], steps=1)
        self._save_predictions(image_batch, predictions, output_dir)

    def _load_combined_model(self, artifact_path: str = "./models/", name: str = "cnn"):
        model_path = os.path.join(artifact_path, "models")
        self.feature_extractor = tf.keras.models.load_model(os.path.join(model_path, name + "_feature_extractor.h5"))
        self.head = tf.keras.models.load_model(os.path.join(model_path, name + "_head.h5"))
        self.model = tf.keras.models.Sequential([self.feature_extractor, self.head])
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.01), # TODO move to config
            metrics=['accuracy'],
        )
        self.model.summary()

    def _create_model(self, model_architecture, model_head, image_size, num_classes):
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
        feature_extractor.add(Dense(self.config["model"]["num_output_features"], activation="relu"))

        if model_head == "deterministic":
            head = Sequential([
                Dense(self.config["model"]["num_output_features"], activation="relu"),
                Dense(int(num_classes), activation="softmax")
            ])
        elif model_head == "bayesian":
            tfd = tfp.distributions
            kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                                      tf.cast(self.n_training_points, dtype=tf.float32))
            head = tf.keras.Sequential([
                tfp.layers.DenseReparameterization(units=self.config["model"]["num_output_features"], kernel_divergence_fn=kl_divergence_function,
                                                   bias_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
                tfp.layers.DenseReparameterization(units=int(num_classes),kernel_divergence_fn=kl_divergence_function,
                                                   bias_divergence_fn=kl_divergence_function, activation="softmax"),
            ])
        elif model_head == "gp":
            num_inducing_points = 50
            head = tf.keras.Sequential([
                tfp.layers.VariationalGaussianProcess(
                num_inducing_points=num_inducing_points,
                kernel_provider=RBFKernelFn(),
                event_shape=[num_classes], # output dimensions
                inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                    minval=0.0, maxval=1.0, seed=None
                ),
                jitter=10e-3
                # unconstrained_observation_noise_variance_initializer=(
                #     tf.constant_initializer(np.array(0.54).astype(np.float32))),
                ),
                tf.keras.layers.Softmax()
            ])
        else:
            raise Exception("Choose valid model head!")
        model = Sequential([feature_extractor, head])
        model.compile(optimizer='adam',
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy'])

        # self.feature_extractor = feature_extractor
        # self.head = head
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


class RBFKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5. * self._length_scale)
        )
