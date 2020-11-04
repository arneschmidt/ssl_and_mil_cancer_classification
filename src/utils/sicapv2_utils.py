import os
import pandas as pd
import numpy as np
from random import sample


def extract_sicap_df_info(dataframe_raw, data_config):
    # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
    dataframe = pd.DataFrame()
    dataframe["image_path"] = 'images/' + dataframe_raw["image_name"]
    dataframe["class"] = np.argmax([dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]],
                                   axis=0).astype(str)
    if data_config['supervision'] == 'mil':
        wsi_df = pd.read_excel(os.path.join(data_config["dir"], "wsi_labels.xlsx"))
        dataframe = adopt_dataframe_to_mil(dataframe, wsi_df, data_config['positive_instance_labels_per_bag'])
    # return dataframe with some instance labels
    return dataframe

def adopt_dataframe_to_mil(dataframe, wsi_dataframe, num_instance_samples):
    dataframe["instance_label"] = 4 # class_id 4: unlabeled
    dataframe["wsi_labels"] = np.NaN
    dataframe["wsi_labels"] = dataframe["wsi_labels"].astype(object)
    wsi_dataframe["wsi_labels"] = np.NaN
    wsi_dataframe["wsi_labels"] = wsi_dataframe["wsi_labels"].astype(object)
    wsi_dataframe['wsi_max_gleason_grade'] = np.max \
        ([wsi_dataframe['Gleason_primary'], wsi_dataframe['Gleason_secondary']], axis=0)
    for wsi_df_row in range(len(wsi_dataframe["wsi_labels"])):
        wsi_dataframe["wsi_labels"][wsi_df_row] = np.arange \
            (np.max(wsi_dataframe['wsi_max_gleason_grade'][wsi_df_row ] -1, 0))
        if wsi_dataframe['Gleason_primary'][wsi_df_row] == wsi_dataframe['Gleason_secondary'][wsi_df_row]  == 0:
            negative_bag = True
        else:
            negative_bag = False
        primary_gleason_grade_rows = []
        secondary_gleason_grade_rows = []
        for patch_df_row in range(len(dataframe["image_path"])):
            if wsi_dataframe['slide_id'][wsi_df_row] in dataframe["image_path"][patch_df_row]:
                dataframe["wsi_labels"][patch_df_row] = wsi_dataframe["wsi_labels"][wsi_df_row]
                if negative_bag:
                    dataframe["instance_label"][patch_df_row] = 0
                elif wsi_dataframe['Gleason_primary'][wsi_df_row] - 2 == int(dataframe["class"][patch_df_row]):
                    primary_gleason_grade_rows.append(patch_df_row)
                elif wsi_dataframe['Gleason_secondary'][wsi_df_row] - 2 == int(dataframe["class"][patch_df_row]):
                    secondary_gleason_grade_rows.append(patch_df_row)
        sampled_primary_rows = sample_or_complete_list(primary_gleason_grade_rows, num_instance_samples)
        sampled_secondary_rows = sample_or_complete_list(secondary_gleason_grade_rows, num_instance_samples)
        if not negative_bag:
            dataframe['instance_label'][sampled_primary_rows] = dataframe['class'][sampled_primary_rows]
            dataframe['instance_label'][sampled_secondary_rows] = dataframe['class'][sampled_secondary_rows]
    dataframe['instance_label'] = dataframe['instance_label']
    return dataframe

def sample_or_complete_list(list, num_samples):
    if num_samples >= len(list):
        return list
    else:
        return sample(list, num_samples)