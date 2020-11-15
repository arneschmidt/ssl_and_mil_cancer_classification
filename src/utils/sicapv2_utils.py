import os
import random
import pandas as pd
import numpy as np


def extract_sicap_df_info(dataframe_raw, data_config, split='train'):
    print('Preparing data split '+split)
    # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
    dataframe = pd.DataFrame()
    dataframe["image_path"] = 'images/' + dataframe_raw["image_name"]

    if data_config['supervision'] == 'mil':
        dataframe["class"] = np.argmax(
            [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"],
             dataframe_raw["unlabeled"]],
            axis=0).astype(str)
        wsi_df = pd.read_excel(os.path.join(data_config["dir"], "wsi_labels.xlsx"))
        dataframe = adopt_dataframe_to_mil(dataframe, wsi_df, data_config['positive_instance_labels_per_bag'], split)
    else:
        dataframe["class"] = np.argmax(
            [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]],
            axis=0).astype(str)

    # return dataframe with some instance labels
    return dataframe

def adopt_dataframe_to_mil(dataframe, wsi_dataframe, num_instance_samples, split='train'):
    dataframe, wsi_dataframe = set_wsi_labels(dataframe, wsi_dataframe)
    if split == 'train':
        rows_of_visible_instance_labels = get_rows_of_visible_instances(dataframe, wsi_dataframe, num_instance_samples)
        dataframe["instance_label"] = 4  # class_id 4: unlabeled
        dataframe["instance_label"][rows_of_visible_instance_labels] = dataframe['class']
        dataframe['class'] = dataframe['instance_label'].astype(str)
    # elif split == 'val':
    #     rows_of_visible_instance_labels = get_rows_of_visible_instances(dataframe, wsi_dataframe, num_instance_samples)
    #     dataframe = dataframe[rows_of_visible_instance_labels]
    else:
        dataframe = dataframe[dataframe["class"].str.match('4') == False]

    return dataframe

def set_wsi_labels(dataframe, wsi_dataframe):
    dataframe["wsi"] = np.NaN
    dataframe["wsi_labels"] = np.NaN
    dataframe["wsi_labels"] = dataframe["wsi_labels"].astype(object)
    wsi_dataframe["wsi_labels"] = np.NaN
    wsi_dataframe["wsi_labels"] = wsi_dataframe["wsi_labels"].astype(object)
    wsi_dataframe['wsi_max_gleason_grade'] = np.max \
        ([wsi_dataframe['Gleason_primary'], wsi_dataframe['Gleason_secondary']], axis=0)
    for wsi_df_row in range(len(wsi_dataframe["slide_id"])):
        for instance_df_row in range(len(dataframe["image_path"])):
            # wsi_dataframe["wsi_labels"][wsi_df_row] = np.arange \
            #     (np.max(wsi_dataframe['wsi_max_gleason_grade'][wsi_df_row] - 1, 0))
            wsi_dataframe["wsi_labels"][wsi_df_row] = np.array([np.max(wsi_dataframe['Gleason_primary'][wsi_df_row] -2,0),
                                                                np.max(wsi_dataframe['Gleason_secondary'][wsi_df_row] -2,0)])
            if wsi_dataframe['slide_id'][wsi_df_row] in dataframe["image_path"][instance_df_row]:
                dataframe["wsi"][instance_df_row] = wsi_dataframe['slide_id'][wsi_df_row]
                dataframe["wsi_labels"][instance_df_row] = wsi_dataframe["wsi_labels"][wsi_df_row]
    return dataframe, wsi_dataframe

def get_rows_of_visible_instances(dataframe, wsi_dataframe, num_instance_samples):
    rows_of_visible_instance_labels = []
    for wsi_df_row in range(len(wsi_dataframe["wsi_labels"])):
        if wsi_dataframe['Gleason_primary'][wsi_df_row] == wsi_dataframe['Gleason_secondary'][wsi_df_row] == 0:
            negative_bag = True
        else:
            negative_bag = False

        primary_gleason_grade_rows = []
        secondary_gleason_grade_rows = []
        for instance_df_row in range(len(dataframe["image_path"])):
            if wsi_dataframe['slide_id'][wsi_df_row] in dataframe["image_path"][instance_df_row]:
                if negative_bag:
                    rows_of_visible_instance_labels.append(instance_df_row)
                elif wsi_dataframe['Gleason_primary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
                    primary_gleason_grade_rows.append(instance_df_row)
                elif wsi_dataframe['Gleason_secondary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
                    secondary_gleason_grade_rows.append(instance_df_row)
        rows_of_visible_instance_labels += sample_or_complete_list(primary_gleason_grade_rows, num_instance_samples)
        rows_of_visible_instance_labels += sample_or_complete_list(secondary_gleason_grade_rows, num_instance_samples)
    return rows_of_visible_instance_labels


def sample_or_complete_list(list, num_samples):
    random.seed(42)
    if num_samples >= len(list):
        return list
    else:
        return random.sample(list, num_samples)