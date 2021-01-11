import argparse
import os
import collections
import yaml
import tensorflow as tf
from typing import Dict, Optional, Tuple
from data import DataGenerator
from supervised_model import SupervisedModel
from mil_model import MILModel
from mlflow_log import MLFlowLogger

def main(config):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    logger = MLFlowLogger(config)
    logger.config_logging()

    print("Create data generators..")
    data_gen = DataGenerator(config["data"], config["model"]["batch_size"], config['model']['mode'])

    print("Load classification model")
    if config['data']['supervision'] == 'full':
        model = SupervisedModel(config, data_gen.num_classes, data_gen.num_training_samples)
    else:
        model = MILModel(config, data_gen.num_classes, data_gen.num_training_samples)

    if config["model"]["mode"] == "train":
        print("Train")
        logger.data_logging(data_gen.get_train_data_statistics())
        model.train(data_gen)
    elif config["model"]["mode"] == "test":
        print("Test")
        metrics = model.test(data_gen)
        logger.test_logging(metrics)
    elif config["model"]["mode"] == "predict":
        print("Predict")
        model.predict(data_gen)
    elif config["model"]["mode"] == "predict_features":
        model.predict_features(data_gen)

    if config['logging']['log_artifacts']:
        logger.log_artifacts()


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

def load_configs(args):
    with open(args.default_config) as file:
        config = yaml.full_load(file)
    with open(config["data"]["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = config_update(config, config_data_dependent)

    if args.experiment_config != 'None':
        with open(args.experiment_config) as file:
            exp_config = yaml.full_load(file)
        config = config_update(config, exp_config)

    return config

if __name__ == "__main__":
    print('Load configuration')
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_config", "-ec", type=str, default="None",
                        help="Config path to experiment config. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    config = load_configs(args)

    print('Create output folder')
    config['output_dir'] = os.path.join(config['data']['artifact_dir'], config['logging']['run_name'])
    os.makedirs(config['output_dir'], exist_ok=True)
    print('Output will be written to: ', config['output_dir'])

    main(config)