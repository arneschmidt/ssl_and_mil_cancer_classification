import argparse
import mlflow
import os
import collections
import yaml
import tensorflow as tf
from configloader import ConfigLoader
from typing import Dict, Optional, Tuple
from src.data import DataGenerator
from src.model import ClassficationModel

def main(config):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    mlflow.set_tracking_uri(os.path.join(config["data"]["artifact_dir"], "mlruns"))
    with mlflow.start_run(experiment_id=mlflow.set_experiment(config["data"]["dataset_name"]), run_name="test_2"):
        print("Create data generators..")
        data_gen = DataGenerator(config["data"], config["model"]["batch_size"])
        train_data, val_data, test_data = data_gen.generate_data()

        print("Load classification model")
        num_classes = len(train_data.class_indices.values())
        classification_model = ClassficationModel(config, num_classes)

        if config["model"]["mode"] == "train":
            print("Train")
            classification_model.train(train_data, val_data, config["model"]["save_name"])
            classification_model.save(config["data"]["artifact_dir"], config["model"]["save_name"])
        elif config["model"]["mode"] == "test":
            print("Test")
            classification_model.test(test_data)
        elif config["model"]["mode"] == "predict":
            print("Predict")
            classification_model.predict(test_data, config["data"]["artifact_dir"])


def update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

if __name__ == "__main__":
    # # TODO: move to config.yaml
    # parser = argparse.ArgumentParser(description="Breast Cancer Classification")
    # parser.add_argument("--mode", "-m", type=str, default="train",
    #                     help="Choose mode: train or test")
    # parser.add_argument("--model_architecture", "-ma", type=str, default="efficientnetb0",
    #                     help="Choose model architecture: mobilenetv2, efficientnetb0 or simple_cnn")
    # parser.add_argument("--dataset_name", "-dn", type=str, default="sicapv2",
    #                     help="Dataset directory.")
    # parser.add_argument("--data_dir", "-dd", type=str, default="/home/arne/datasets/SICAPv2/images/",
    #                     help="Dataset directory.")
    # parser.add_argument("--artifact_base_folder", "-a", type=str, default="./dataset_dependent/sicapv2",
    #                     help="Dataset directory.")
    # parser.add_argument("--data_split_dir", "-f", type=str, default="/home/arne/datasets/SICAPv2/partition/Validation/Val1/",
    #                     help="path to the directory of data lists")
    # parser.add_argument("--save_model", "-s", type=str, default="models/simple_cnn.h5",
    #                     help="Path to save model.")
    # parser.add_argument("--load_model", "-l", type=str, default="None",
    #                     help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    # parser.add_argument("--output_dir", "-o", type=str, default="output/",
    #                     help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    # parser.add_argument("--batch_size", "-b", type=int, default=32,
    #                     help="Batch size")
    # parser.add_argument("--image_target_size", "-i", type=Tuple, default=(250, 250),
    #                     help="Image resolution after rescaling")
    # args = parser.parse_args()
    with open(r'./config.yaml') as file:
        config = yaml.full_load(file)
    with open(config["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = update(config, config_data_dependent)
    main(config)