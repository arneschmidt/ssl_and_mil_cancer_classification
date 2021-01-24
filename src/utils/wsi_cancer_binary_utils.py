import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score

def calc_wsi_cancer_binary_metrics(wsi_predict_dataframe, wsi_gt_dataframe):
    # TODO: calc metric and cofusion matrices
    metrics = {}
    predictions = np.array(wsi_predict_dataframe['class'])
    confidences = np.array(wsi_predict_dataframe['confidence'])
    gt = np.array(wsi_gt_dataframe['class'])
    metrics['wsi_accuracy'] = accuracy_score(gt, predictions)
    metrics['wsi_cohens_quadratic_kappa'] = cohen_kappa_score(gt, predictions, weights='quadratic')
    metrics['wsi_f1_score'] = f1_score(gt, predictions)
    metrics['wsi_auc'] = roc_auc_score(gt, confidences)
    roc = {}
    roc['fpr'], roc['tpr'], roc['thresholds'] = roc_curve(gt, confidences)
    artifacts = {}
    artifacts['roc'] = roc

    value_to_be_optimized = None
    return metrics, artifacts, value_to_be_optimized

def calc_wsi_binary_prediction(num_predictions_per_class, confidences_per_class):
    if num_predictions_per_class[1] > 5 and confidences_per_class[1] > 0.5:
        class_pred = 1
    else:
        class_pred = 0
    confidence = confidences_per_class[1]
    assert confidence >= 0.0 and confidence <= 1.0
    return class_pred, confidence