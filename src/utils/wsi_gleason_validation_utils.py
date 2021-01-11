import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def get_wsi_gleason_metrics(model, data_gen, patch_dataframe, wsi_dataframe, batch_size, confidence_threshold=0.6, num_patch_threshold=3):
    predictions = model.predict(data_gen, batch_size=batch_size, steps=np.ceil(data_gen.n / batch_size), verbose=1)
    wsi_predict_dataframe = get_predictions_per_wsi(patch_dataframe, predictions, confidence_threshold, num_patch_threshold)
    wsi_gt_dataframe = wsi_dataframe[wsi_dataframe['slide_id'].isin(wsi_predict_dataframe['slide_id'])]

    wsi_predict_dataframe = get_gleason_score_and_isup_grade(wsi_predict_dataframe)
    wsi_gt_dataframe = get_gleason_score_and_isup_grade(wsi_gt_dataframe)

    metrics_dict = {}

    metrics_dict['wsi_gs_cohens_quadratic_kappa'] = cohen_kappa_score(wsi_predict_dataframe['gleason_score'],
                                                                 wsi_gt_dataframe['gleason_score'],
                                                                 weights='quadratic')
    metrics_dict['wsi_isup_cohens_quadratic_kappa'] = cohen_kappa_score(wsi_predict_dataframe['isup_grade'],
                                                                   wsi_gt_dataframe['isup_grade'],
                                                                   weights='quadratic')
    confusion_matrices = {}
    confusion_matrices['wsi_gs_confusion_matrix'] = confusion_matrix(wsi_predict_dataframe['gleason_score'],
                                                                     wsi_gt_dataframe['gleason_score'],
                                                                     labels=[0, 6, 7, 8, 9, 10])
    confusion_matrices['wsi_isup_confusion_matrix'] = confusion_matrix(wsi_predict_dataframe['isup_grade'],
                                                                       wsi_gt_dataframe['isup_grade'],
                                                                       labels=[0, 1, 2, 3, 4, 5])
    return metrics_dict, confusion_matrices


def get_predictions_per_wsi(patch_dataframe, predictions_softmax, confidence_threshold, num_patch_threshold):
    confidences = np.max(predictions_softmax, axis=1)
    predictions = np.argmax(predictions_softmax, axis=1)
    wsi_names = []
    wsi_primary = []
    wsi_secondary = []
    num_predictions_per_class = [0, 0, 0, 0]

    row = 0
    while True:
        wsi_name = patch_dataframe['wsi'][row]
        wsi_df = patch_dataframe[patch_dataframe['wsi'] == wsi_name]
        end_row_wsi = row + len(wsi_df)
        for class_id in range(len(num_predictions_per_class)):
            class_id_predicted = np.logical_and(predictions[row:end_row_wsi] == class_id,
                                                confidences[row:end_row_wsi] > confidence_threshold)
            num_predictions = np.count_nonzero(class_id_predicted)
            # filter out outliers
            if num_predictions < num_patch_threshold:
                num_predictions_per_class[class_id] = 0
            else:
                num_predictions_per_class[class_id] = num_predictions
            # not cancerous
        if num_predictions_per_class[1] == num_predictions_per_class[2] == num_predictions_per_class[3] == 0:
            primary = 0
            secondary = 0
        # only one gleason grade
        elif num_predictions_per_class[2] == num_predictions_per_class[3] == 0:
            primary = 3
            secondary = 3
        elif num_predictions_per_class[1] == num_predictions_per_class[3] == 0:
            primary = 4
            secondary = 4
        elif num_predictions_per_class[1] == num_predictions_per_class[2] == 0:
            primary = 5
            secondary = 5
        # two gleason grades
        else:
            # Make sure we don't include class 0 here. Argsort returns value 0,1 or 2. Add 3 to get gleason grade.
            primary = np.argsort(num_predictions_per_class[1:4])[2] + 3
            secondary = np.argsort(num_predictions_per_class[1:4])[1] + 3

        wsi_names.append(wsi_name)
        wsi_primary.append(primary)
        wsi_secondary.append(secondary)
        if end_row_wsi == len(patch_dataframe):
            break
        else:
            row = end_row_wsi

    wsi_predict_dataframe = pd.DataFrame()
    wsi_predict_dataframe['slide_id'] = wsi_names
    wsi_predict_dataframe['Gleason_primary'] = wsi_primary
    wsi_predict_dataframe['Gleason_secondary'] = wsi_secondary

    return wsi_predict_dataframe


def get_gleason_score_and_isup_grade(wsi_df):
    wsi_df['gleason_score'] = wsi_df[['Gleason_primary', 'Gleason_secondary']].sum(axis=1)
    isup_grade = np.full(shape=len(wsi_df), fill_value=-1)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 0, wsi_df['Gleason_secondary'] == 0), 0, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 3, wsi_df['Gleason_secondary'] == 3), 1, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 3, wsi_df['Gleason_secondary'] == 4), 2, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 4, wsi_df['Gleason_secondary'] == 3), 3, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 3, wsi_df['Gleason_secondary'] == 5), 4, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 4, wsi_df['Gleason_secondary'] == 4), 4, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 5, wsi_df['Gleason_secondary'] == 3), 4, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 4, wsi_df['Gleason_secondary'] == 5), 5, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 5, wsi_df['Gleason_secondary'] == 4), 5, isup_grade)
    isup_grade = np.where(np.logical_and(wsi_df['Gleason_primary'] == 5, wsi_df['Gleason_secondary'] == 5), 5, isup_grade)
    assert(np.all(isup_grade >= 0))

    wsi_df['isup_grade'] = isup_grade

    return wsi_df





