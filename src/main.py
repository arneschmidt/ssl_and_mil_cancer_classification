import argparse
import os
import collections
import yaml
import tensorflow as tf
from typing import Dict, Optional, Tuple
from src.data import DataGenerator
from src.model import ClassficationModel
from src.mlflow_log import MLFlowLogger

def main(config):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    if config["logging"]["log_experiment"]:
        logger = MLFlowLogger(config)
        logger.config_logging()

    print("Create data generators..")
    data_gen = DataGenerator(config["data"], config["model"]["batch_size"])
    train_data, val_data, test_data = data_gen.generate_data()

    print("Load classification model")
    num_classes = len(train_data.class_indices.values())
    classification_model = ClassficationModel(config, num_classes, train_data.n)

    if config["model"]["mode"] == "train":
        print("Train")
        classification_model.train(train_data, val_data)
    elif config["model"]["mode"] == "test":
        print("Test")
        metrics = classification_model.test(test_data)
        if config["logging"]["log_experiment"]:
            logger.test_logging(metrics)
    elif config["model"]["mode"] == "predict":
        print("Predict")
        classification_model.predict(test_data, config["data"]["artifact_dir"])


def config_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = config_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

if __name__ == "__main__":
    with open(r'./config.yaml') as file:
        config = yaml.full_load(file)
    with open(config["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = config_update(config, config_data_dependent)
    main(config)