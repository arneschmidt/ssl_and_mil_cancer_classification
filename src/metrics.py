import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics, calc_gleason_grade
from utils.wsi_cancer_binary_utils import calc_wsi_binary_prediction, calc_wsi_cancer_binary_metrics


class MetricCalculator():
    """
    Object that takes care of the metric calculation for training and testing.
    """
    def __init__(self, model, data_gen, config, mode):
        self.model = model
        self.data_gen = data_gen
        self.mode = mode
        self.dataset_type = config['data']['dataset_type']
        self.num_classes = config['data']['num_classes']
        self.metrics_patch_level = config['model']['metrics_patch_level']
        self.metrics_wsi_level = config['model']['metrics_wsi_level']
        if mode == 'val':
            self.val_gen = data_gen.validation_generator
            self.val_df = data_gen.val_df
            self.test_gen = self.val_gen
            self.test_df = self.val_df
        else:
            self.val_gen = data_gen.validation_generator
            self.val_df = data_gen.val_df
            self.test_gen = data_gen.test_generator
            self.test_df = data_gen.test_df

    def calc_metrics(self):
        """
        Make predictions and calculate metrics.
        :return: metrics dict, artifacts dict
        """
        print('\nCalculate metrics for ' + self.mode)
        val_predictions, test_predictions = self.get_predictions()
        metrics = {}
        artifacts = {}
        if self.metrics_patch_level:
            metrics.update(self.calc_patch_level_metrics(test_predictions))
        if self.metrics_wsi_level:
            wsi_metrics, artifacts = self.calc_optimal_wsi_metrics(val_predictions, test_predictions)
            metrics.update(wsi_metrics)
        metrics = self.add_prefix(metrics, self.mode)
        # for key in artifacts.keys():
        #     artifacts[key] = self.add_prefix(artifacts[key], self.mode)
        print('Metrics ' + self.mode)
        print(metrics)
        return metrics, artifacts

    def get_predictions(self):
        """
        Obtain model predictions for validation and/or test data.
        :return: val_predictions, test_predictions
        """
        model = self.model
        data_gen = self.data_gen
        batch_size = data_gen.validation_generator.batch_size
        if self.mode == 'val':
            val_gen = self.val_gen
            val_predictions = model.predict(val_gen, batch_size=batch_size, steps=np.ceil(val_gen.n / batch_size), verbose=1)
            test_predictions = val_predictions
        else:
            val_gen = self.val_gen
            val_predictions = model.predict(val_gen, batch_size=batch_size,
                                                    steps=np.ceil(val_gen.n / batch_size), verbose=1)
            test_gen = self.test_gen
            test_predictions = model.predict(test_gen, batch_size=batch_size, steps=np.ceil(test_gen.n / batch_size), verbose=1)

        return val_predictions, test_predictions

    def calc_patch_level_metrics(self, predictions_softmax: np.array):
        """
        Calculate the metrics for the instance (patch) level.
        """
        predictions = np.argmax(predictions_softmax, axis=1)
        unlabeled_index = self.num_classes
        gt_classes = self.test_df['class']
        indices_of_labeled_patches = (gt_classes != str(unlabeled_index))
        gt_classes = np.array(gt_classes[indices_of_labeled_patches]).astype(np.int)
        predictions = np.array(predictions[indices_of_labeled_patches]).astype(np.int)

        metrics ={}
        metrics['accuracy'] = accuracy_score(gt_classes, predictions)
        metrics['cohens_quadratic_kappa'] = cohen_kappa_score(gt_classes, predictions, weights='quadratic')
        metrics['f1_mean'] = f1_score(gt_classes, predictions, average='macro')
        f1_score_classwise = f1_score(gt_classes, predictions, average=None)
        for class_id in range(len(f1_score_classwise)):
            key = 'f1_class_id_' + str(class_id)
            metrics[key] = f1_score_classwise[class_id]
        return metrics

    def calc_optimal_wsi_metrics(self, val_predictions: np.array, test_predictions: np.array):
        """
        Calculate the WSI metrics for an optimal instance-level confidence threshold.
        We filter out outliers by disregarding predictions below an optimal threshold to avoid false positive bag
        predictions.
        :param val_predictions: 
        :param test_predictions:
        :return:
        """
        confidence_threshold = self.calc_optimal_confidence_threshold(val_predictions, self.val_df)
        metrics_dict, artifacts, _ = self.calc_wsi_metrics(test_predictions, self.test_df, confidence_threshold, 'test')
        metrics_dict['confidence_threshold'] = confidence_threshold
        return metrics_dict, artifacts

    def calc_wsi_metrics(self, predictions: np.array, gt_df, confidence_threshold: float, name: str):
        """
        Calculate the metrics for all WSI (bag-level)
        """
        wsi_dataframe = self.data_gen.wsi_df
        wsi_predict_dataframe = self.get_predictions_per_wsi(predictions, gt_df, confidence_threshold)
        wsi_gt_dataframe = wsi_dataframe[wsi_dataframe['slide_id'].isin(wsi_predict_dataframe['slide_id'])]
        wsi_predict_dataframe, wsi_gt_dataframe = self.sort_dataframes(wsi_predict_dataframe, wsi_gt_dataframe)
        # wsi_gt_dataframe.to_csv(name + '_wsi_gt_dataframe.csv')
        # wsi_predict_dataframe.to_csv(name + '_wsi_predict_dataframe.csv')
        if self.dataset_type == 'prostate_cancer':
            metrics_dict, artifacts, optimization_value = calc_wsi_prostate_cancer_metrics(wsi_predict_dataframe, wsi_gt_dataframe)
        else:
            metrics_dict, artifacts, optimization_value = calc_wsi_cancer_binary_metrics(wsi_predict_dataframe, wsi_gt_dataframe)

        return metrics_dict, artifacts, optimization_value

    def calc_optimal_confidence_threshold(self, predictions: np.array, gt_dataframe: pd.DataFrame):
        """
        Find the optimal threshold based on the validation threshold.
        :param predictions:
        :param gt_dataframe:
        :return:
        """
        confidence_thresholds = np.arange(0.3, 1.0, 0.02)
        best_result = 0.0
        optimal_threshold = 0.0
        for i in range(len(confidence_thresholds)):
            _, _, opt_value = self.calc_wsi_metrics(predictions, gt_dataframe, confidence_thresholds[i], 'val')
            if opt_value > best_result:
                best_result = opt_value
                optimal_threshold = confidence_thresholds[i]
        return optimal_threshold

    def get_predictions_per_wsi(self, predictions_softmax: np.array, patch_dataframe: pd.DataFrame,
                                confidence_threshold: float):
        """
        Get the globale predictions per WSI.
        """
        confidences = np.max(predictions_softmax, axis=1)
        predictions = np.argmax(predictions_softmax, axis=1)
        wsi_names = []
        wsi_primary = []
        wsi_secondary = []
        num_predictions_per_class = np.zeros(shape=predictions_softmax.shape[1])
        confidences_per_class = np.zeros(shape=predictions_softmax.shape[1])

        row = 0
        while True:
            wsi_name = patch_dataframe['wsi'][row]
            wsi_df = patch_dataframe[patch_dataframe['wsi'] == wsi_name]
            end_row_wsi = row + len(wsi_df)
            for class_id in range(len(num_predictions_per_class))[1:]:
                predictions_for_wsi = predictions[row:end_row_wsi]
                confidences_for_wsi = confidences[row:end_row_wsi]
                class_id_predicted = (predictions_for_wsi == class_id)
                # top_5_confidences = np.sort(confidences_for_wsi[class_id_predicted], axis=0)[::-1][0:5]
                # if top_5_confidences.size == 0:
                #     top_5_conf_average = 0.0
                # else:
                #     top_5_conf_average = top_5_confidences[0] #np.sum(top_5_confidences)/5
                if confidences_for_wsi[class_id_predicted].size == 0:
                    top_confidence = 0.0
                else:
                    top_confidence = np.max(confidences_for_wsi[class_id_predicted], axis=0)

                class_id_predicted_with_confidence = confidences_for_wsi[class_id_predicted] > confidence_threshold
                num_predictions = np.count_nonzero(class_id_predicted_with_confidence)
                num_predictions_per_class[class_id] = num_predictions
                confidences_per_class[class_id] = top_confidence

            if self.dataset_type == 'prostate_cancer':
                primary, secondary = calc_gleason_grade(num_predictions_per_class, confidences_per_class, confidence_threshold)
            else: # NOTE: here we use primary=class, secondary=confidence
                primary, secondary = calc_wsi_binary_prediction(num_predictions_per_class, confidences_per_class, confidence_threshold)
            wsi_names.append(wsi_name)
            wsi_primary.append(primary)
            wsi_secondary.append(secondary)
            if end_row_wsi == len(patch_dataframe):
                break
            else:
                row = end_row_wsi
        assert len(wsi_names) == len(wsi_primary) == len(wsi_secondary)

        wsi_primary = np.array(wsi_primary)
        wsi_secondary = np.array(wsi_secondary)

        wsi_predict_dataframe = pd.DataFrame()
        wsi_predict_dataframe['slide_id'] = np.array(wsi_names)
        if self.dataset_type == 'prostate_cancer':
            wsi_predict_dataframe['Gleason_primary'] = wsi_primary
            wsi_predict_dataframe['Gleason_secondary'] = wsi_secondary
        else:
            wsi_predict_dataframe['class'] = wsi_primary
            wsi_predict_dataframe['confidence'] = wsi_secondary
            assert np.all(np.logical_or(wsi_primary == 0, wsi_primary == 1))
            assert np.all(np.logical_and(wsi_secondary >= 0.0, wsi_secondary <= 1.0))

        return wsi_predict_dataframe

    def sort_dataframes(self, wsi_predict_dataframe: pd.DataFrame, wsi_gt_dataframe: pd.DataFrame):
        wsi_predict_dataframe = wsi_predict_dataframe.sort_values(by='slide_id', inplace=False)
        wsi_gt_dataframe = wsi_gt_dataframe.sort_values(by='slide_id', inplace=False)
        if not len(wsi_predict_dataframe) == len(wsi_gt_dataframe):
            wsi_gt_dataframe.to_csv('wsi_gt_dataframe.csv')
            wsi_predict_dataframe.to_csv('wsi_predict_dataframe.csv')
            raise Warning('Dataframes do not have the same length! Saving for debugging.')
        return wsi_predict_dataframe, wsi_gt_dataframe

    def add_prefix(self, dict, prefix):
        new_dict = {}
        for key in dict.keys():
            new_key = prefix + '_' + key
            new_dict[new_key] = dict[key]
        return new_dict