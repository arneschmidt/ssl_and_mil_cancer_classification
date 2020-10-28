import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, GlobalMaxPool1D, GlobalMaxPool2D, SeparableConv2D


def create_model(config, num_classes, num_training_points):
    feature_extractor = create_feature_extactor(config)
    head = create_head(config, num_classes, num_training_points)

    model = Sequential([feature_extractor, head])
    return model

def create_feature_extactor(config):
    input_shape = (config["data"]["image_target_size"][0], config["data"]["image_target_size"][1], 3)
    feature_extractor_type = config["model"]["feature_extractor"]["type"]

    weights = "imagenet"
    feature_extractor = Sequential()
    if feature_extractor_type == "mobilenetv2":
        feature_extractor.add(MobileNetV2(include_top=False, input_shape=input_shape, weights=None, pooling='avg'))
    elif feature_extractor_type == "efficientnetb0":
        feature_extractor.add(EfficientNetB0(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "efficientnetb1":
        feature_extractor.add(EfficientNetB1(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "resnet50":
        feature_extractor.add(ResNet50(include_top=False, input_shape=input_shape, weights=weights, pooling='avg'))
    elif feature_extractor_type == "simple_cnn":
        feature_extractor.add(tf.keras.layers.Input(shape=input_shape))
        feature_extractor.add(SeparableConv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        for i in range(3):
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
            feature_extractor.add(MaxPool2D(pool_size=(2, 2)))
        feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
        feature_extractor.add(SeparableConv2D(32, kernel_size=3, activation='relu'))
    elif feature_extractor_type == "fsconv":
        feature_extractor.add(tf.keras.layers.Input(shape=input_shape))
        feature_extractor.add(Conv2D(32, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
        feature_extractor.add(Conv2D(124, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
        feature_extractor.add(Conv2D(512, kernel_size=3, activation='relu'))
        feature_extractor.add(MaxPool2D(strides=(2, 2)))
        # feature_extractor.add(Flatten())
    else:
        raise Exception("Choose valid model architecture!")

    if config["model"]["feature_extractor"]["global_max_pooling"]:
        feature_extractor.add(GlobalMaxPool2D())
    if config["model"]["feature_extractor"]["num_output_features"] > 0:
        activation = config["model"]["feature_extractor"]["output_activation"]
        feature_extractor.add(Dense(config["model"]["feature_extractor"]["num_output_features"], activation=activation))
    # feature_extractor.build(input_shape=input_shape)
    return feature_extractor


def create_head(config, num_classes, num_training_points):
    head_type = config["model"]["head"]["type"]
    mode = config["model"]["mode"]
    if head_type == "deterministic":
        hidden_units = config["model"]["head"]["deterministic"]["number_hidden_units"]
        dropout_rate = config["model"]["head"]["deterministic"]["dropout"]
        head = Sequential()
        head.add(Dropout(rate=dropout_rate))
        if hidden_units > 0 :
            head.add(Dense(hidden_units, activation="relu"))
        head.add(Dense(int(num_classes), activation="softmax"))

    elif head_type == "bnn":
        number_hidden_units = config["model"]["head"]["bnn"]["number_hidden_units"]
        kl_factor = config["model"]["head"]["bnn"]["kl_loss_factor"]
        tfd = tfp.distributions
        # scaling of KL divergence to batch is included already, scaling to dataset size needs to be done
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p)* kl_factor /  # pylint: disable=g-long-lambda
                                                  tf.cast(num_training_points, dtype=tf.float32))

        if mode == 'test':
            tensor_fn = (lambda d: d.mean())
        else:
            tensor_fn = (lambda d: d.sample())

        head = tf.keras.Sequential([
            tfp.layers.DenseReparameterization(activation=tf.nn.relu, units=number_hidden_units,
                                               kernel_divergence_fn=kl_divergence_function,
                                               bias_divergence_fn=kl_divergence_function,
                                               kernel_posterior_tensor_fn=tensor_fn,
                                               bias_posterior_tensor_fn=tensor_fn
                                               ),
            tfp.layers.DenseReparameterization(activation="softmax", units=int(num_classes),
                                               kernel_divergence_fn=kl_divergence_function,
                                               bias_divergence_fn=kl_divergence_function,
                                               kernel_posterior_tensor_fn=tensor_fn,
                                               bias_posterior_tensor_fn=tensor_fn
                                               ),
        ])
    elif head_type == "gp":
        num_inducing_points = config["model"]["head"]["gp"]["inducing_points"]
        features = config["model"]["feature_extractor"]["num_output_features"]

        if mode == 'test':
            tensor_fn = tfp.distributions.Distribution.mean
        else:
            tensor_fn = tfp.distributions.Distribution.sample
        if features < 1:
            raise Exception('Please set the num_output_features > 0 when using Gaussian processes.')
        head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[features]), #, batch_size=config["model"]["batch_size"]),
            tfp.layers.VariationalGaussianProcess(
                num_inducing_points=num_inducing_points,
                kernel_provider=RBFKernelFn(),
                event_shape=[num_classes], # output dimensions
                inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                    minval=0.0, maxval=1.0, seed=None
                ),
                jitter=10e-3,
                convert_to_tensor_fn=tensor_fn,
                # unconstrained_observation_noise_variance_initializer=(
                #     tf.constant_initializer(np.array(0.54).astype(np.float32))),
            ),
            tf.keras.layers.Softmax()
        ])
        # scaling KL divergence to batch size and dataset size
        kl_weight = np.array(config["model"]["batch_size"], np.float32) / num_training_points
        head.add_loss(tf.reduce_sum(kl_weight * head.layers[0].submodules[5].surrogate_posterior_kl_divergence_prior()))
        head.build()
    else:
        raise Exception("Choose valid model head!")
    return head


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
            amplitude=tf.nn.softplus(1.0 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(1.0 * self._length_scale) # 5.
        )
