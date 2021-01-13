from typing import Dict
import mlflow
import tensorflow
import os
import numpy as np

class MLFlowLogger:
    def __init__(self, config: Dict):
        mlflow.set_tracking_uri(config["logging"]["tracking_url"])
        experiment_id = mlflow.set_experiment(experiment_name=config["data"]["dataset_name"])
        mlflow.start_run(experiment_id=experiment_id, run_name=config["logging"]["run_name"])
        self.config = config

    def config_logging(self):
        mlflow.log_params(self.config['model'])
        mlflow.log_params(self.config['model']['feature_extractor'])
        mlflow.log_params(self.config['data'])
        head_type = self.config['model']['head']['type']
        mlflow.log_param('head_type', head_type)
        mlflow.log_params(self.config['model']['head'][head_type])

    def data_logging(self, data_dict):
        mlflow.log_params(data_dict)

    def test_logging(self, metrics: Dict):
        mlflow.log_metrics(metrics)

    def log_artifacts(self):
        mlflow.log_artifacts(self.config['output_dir'])


class MLFlowCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, config, metric_calculator):
        super().__init__()
        self.finished_epochs = 0
        self.best_result = 0.0
        self.new_best_result = False
        self.config = config
        self.metric_calculator = metric_calculator

    def on_batch_end(self, batch: int, logs=None):
        if batch % 100 == 0:
            current_step = int((self.finished_epochs * self.params['steps']) + batch)
            metrics_dict = format_metrics_for_mlflow(logs.copy())
            mlflow.log_metrics(metrics_dict, step=current_step)

    def on_epoch_end(self, epoch: int, logs=None):
        metrics_dict, _ = self.metric_calculator.calc_metrics()
        self.finished_epochs = epoch + 1
        current_step = int(self.finished_epochs * self.params['steps'])

        mlflow.log_metrics(metrics_dict, step=current_step)
        mlflow.log_metric('finished_epochs', self.finished_epochs, step=current_step)

        # Check if new best model
        metrics_for_model_saving = self.config['model']['metrics_for_model_saving']
        if metrics_dict[metrics_for_model_saving] > self.best_result:
            self.new_best_result = True
            print("\n New best model! Saving model..")
            self.best_result = metrics_dict[metrics_for_model_saving]
            if self.config["model"]["save_model"]:
                self._save_model()
            mlflow.log_metric("best_" + metrics_for_model_saving, metrics_dict[metrics_for_model_saving])
            mlflow.log_metric("saved_model_epoch", self.finished_epochs)
        else:
            self.new_best_result = False

    def _save_model(self):
        save_dir = os.path.join(self.config["output_dir"], "models")
        os.makedirs(save_dir, exist_ok=True)
        fe_path = os.path.join(save_dir, "feature_extractor.h5")
        head_path = os.path.join(save_dir, "head.h5")
        self.model.layers[0].save_weights(fe_path)
        self.model.layers[1].save_weights(head_path)


def format_metrics_for_mlflow(metrics_dict):
    # for now, just format f1 score which comes in as an array
    metrics_name = 'f1_score'
    if 'val_f1_score' in metrics_dict.keys():
        prefixes = ['val_', '']
    else:
        prefixes = ['']
    for prefix in prefixes:
        f1_score = metrics_dict.pop(prefix + metrics_name)
        for class_id in range(len(f1_score)):
            key = prefix + 'f1_class_id_' + str(class_id)
            metrics_dict[key] = f1_score[class_id]

        metrics_dict[prefix + 'f1_mean'] = np.mean(f1_score)

    return metrics_dict
