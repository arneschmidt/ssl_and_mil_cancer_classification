import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def get_wsi_gleason_metrics(model, data_gen, patch_dataframe, wsi_dataframe, batch_size):
    predictions = model.predict(data_gen, batch_size=batch_size, steps=np.ceil(data_gen.n / batch_size))
    wsi_predict_dataframe = get_predictions_per_wsi(patch_dataframe, predictions)
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
    # metrics_dict['wsi_gs_confusion_matrix'] = confusion_matrix(wsi_predict_dataframe['gleason_score'],
    #                                                       wsi_gt_dataframe['gleason_score'])
    # metrics_dict['wsi_isup_confusion_matrix'] = confusion_matrix(wsi_predict_dataframe['isup_grade'],
    #                                                         wsi_gt_dataframe['isup_grade'])
    return metrics_dict


def get_predictions_per_wsi(patch_dataframe, predictions):
    predictions = np.argmax(predictions, axis=1)
    wsi_names = []
    wsi_primary = []
    wsi_secondary = []
    num_predictions_per_class = [0, 0, 0, 0]

    row = 0
    while True:
        wsi_name = patch_dataframe['wsi'][row]
        wsi_df = patch_dataframe[patch_dataframe['wsi'].str.match(wsi_name)]
        end_row_wsi = row + len(wsi_df)
        for class_id in range(len(num_predictions_per_class)):
            num_predictions_per_class[class_id] = np.count_nonzero(predictions[row:end_row_wsi] == class_id)
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
            primary = np.argsort(num_predictions_per_class)[3] + 2
            secondary = np.argsort(num_predictions_per_class)[2] + 2

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





