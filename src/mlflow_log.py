from typing import Dict
import mlflow
import tensorflow
import os

class MLFlowLogger:
    def __init__(self, config: Dict):
        mlflow.set_tracking_uri(config["logging"]["tracking_url"])
        experiment_id = mlflow.set_experiment(experiment_name=config["data"]["dataset_name"])
        mlflow.start_run(experiment_id=experiment_id, run_name=config["logging"]["run_name"])
        self.config = config

    def config_logging(self):
        mlflow.log_params(self.config['model'])
        mlflow.log_params(self.config['data'])

    def test_logging(self, metrics: Dict):
        mlflow.log_metrics(metrics)


class MLFlowCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, config):
        super().__init__()
        self.finished_epochs = 0
        self.best_result = 0.0
        self.config = config

    def on_batch_end(self, batch: int, logs=None):
        if batch % 100 == 0:
            current_step = (self.finished_epochs * self.params['steps']) + batch
            mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=current_step)
            mlflow.log_metric('train_loss', logs.get('loss'), step=current_step)

    def on_epoch_end(self, epoch: int, logs=None):
        self.finished_epochs = epoch + 1
        current_step = self.finished_epochs * self.params['steps']

        metrics_dict = {"val_loss": logs["val_loss"], "val_accuracy": logs["val_accuracy"]}

        # Check if new best model
        if logs["val_accuracy"] > self.best_result:
            print("\n New best model! Saving model..")
            self.best_result = logs["val_accuracy"]
            self.save()
            mlflow.log_metric("best_val_accuracy", logs["val_accuracy"])
            mlflow.log_metric("saved_model_epoch", self.finished_epochs)

        mlflow.log_metrics(metrics_dict, step=current_step)
        mlflow.log_metric('finished_epochs', self.finished_epochs, step=current_step)

    def save(self):
        save_dir = os.path.join(self.config["data"]["artifact_dir"], "models")
        name = self.config["model"]["save_name"]
        os.makedirs(save_dir, exist_ok=True)
        fe_path = os.path.join(save_dir, name + "_feature_extractor.h5")
        head_path = os.path.join(save_dir, name + "_head.h5")
        self.model.layers[0].save(fe_path)
        self.model.layers[1].save(head_path)
