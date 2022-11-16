import os
import random
import pandas as pd
import numpy as np


def extract_df_info(dataframe_raw, wsi_df, data_config, split='train', wsi_delimiter='_'):
    print('Preparing data split '+split)
    # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
    dataframe = pd.DataFrame()
    dataframe["image_path"] = 'images/' + dataframe_raw["image_name"]
    wsis = dataframe_raw["image_name"].str.split(wsi_delimiter).str[0]
    dataframe["wsi"] = wsis
    # dataframe_raw["wsi"]  = wsis

    dataframe = get_instance_classes(dataframe, dataframe_raw, wsi_df, data_config, split)

    dataframe = dataframe.sort_values(by=['image_path'], ignore_index=True)
    dataframe = dataframe.reset_index(inplace=False)

    # return dataframe with some instance labels
    return dataframe

def get_instance_classes(dataframe, dataframe_raw, wsi_df, data_config, split):
    if data_config['supervision'] == 'supervised':
        if data_config['dataset_type'] == 'cancer_binary':
            class_columns = [dataframe_raw['N'], dataframe_raw['P']]
        elif data_config['dataset_type'] == 'prostate_cancer':
            class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]]
    else:
        if data_config['dataset_type'] == 'cancer_binary':
            class_columns = [dataframe_raw['N'], dataframe_raw['P'], dataframe_raw['unlabeled']]
        elif data_config['dataset_type'] == 'prostate_cancer':
            class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"],
                             dataframe_raw["unlabeled"]]
    #
    # if data_config['supervision'] == 'mil' and data_config['dataset_type'] == 'prostate_cancer':
    #     class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"], dataframe_raw["unlabeled"]]
    # elif data_config['supervision'] == 'supervised' and data_config['dataset_type'] == 'prostate_cancer':
    #     class_columns = [dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]]
    # elif data_config['supervision'] == 'mil' and data_config['dataset_type'] == 'cancer_binary':
    #     class_columns = [dataframe_raw['N'], dataframe_raw['P'], dataframe_raw['unlabeled']]
    # elif data_config['supervision'] == 'supervised' and data_config['dataset_type'] == 'cancer_binary':
    #     class_columns = [dataframe_raw['N'], dataframe_raw['P']]
    dataframe["class"] = np.argmax(class_columns, axis=0).astype(str)

    if data_config['supervision'] == 'supervised':
        dataframe["wsi_contains_unlabeled"] = False
    else:
        dataframe = adopt_dataframe_to_mil(dataframe, wsi_df, data_config, split)

    return dataframe


def adopt_dataframe_to_mil(dataframe, wsi_dataframe, data_config, split='train'):
    if data_config['dataset_type'] == 'prostate_cancer':
        dataframe = set_wsi_labels_pc(dataframe, wsi_dataframe)
    else:
        dataframe = set_wsi_labels_cb(dataframe, wsi_dataframe)
    if split == 'train':
        num_instance_samples = data_config['positive_instance_labels_per_bag']
        if num_instance_samples != 'all': # in this case we want to use all labels, without masking
            if data_config['dataset_type'] == 'prostate_cancer':
                dataframe = hide_instance_labels_pc(dataframe, wsi_dataframe, num_instance_samples)
            else: # cancer binary = cb
                dataframe = hide_instance_labels_cb(dataframe, wsi_dataframe, num_instance_samples)
        dataframe = check_if_wsi_contains_unlabeled(dataframe, wsi_dataframe, data_config['dataset_type'])

    return dataframe

def hide_instance_labels_pc(dataframe, wsi_dataframe, num_instance_samples):
    rows_of_visible_instance_labels = get_rows_of_visible_instances_pc(dataframe, wsi_dataframe, num_instance_samples)
    dataframe["instance_label"] = 4  # class_id 4: unlabeled
    dataframe["instance_label"][rows_of_visible_instance_labels] = dataframe['class']
    dataframe['class'] = dataframe['instance_label'].astype(str)
    return dataframe

def hide_instance_labels_cb(dataframe, wsi_dataframe, num_instance_samples):
    rows_of_visible_instance_labels = get_rows_of_visible_instances_cb(dataframe, wsi_dataframe, num_instance_samples)
    dataframe["instance_label"] = 2  # class_id 2: unlabeled
    dataframe["instance_label"][rows_of_visible_instance_labels] = dataframe['class']
    dataframe['class'] = dataframe['instance_label'].astype(str)
    return dataframe

# TODO: adapt to binary
def set_wsi_labels_pc(dataframe, wsi_dataframe):
    dataframe['wsi_index'] = -1
    dataframe["wsi_primary_label"] = -1
    dataframe["wsi_secondary_label"] = -1
    dataframe['wsi_contains_unlabeled'] = True
    for row in range(len(wsi_dataframe)):
        id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
        if np.any(id_bool) == False:
            continue
        dataframe['wsi_index'][id_bool] = row
        dataframe['wsi_primary_label'][id_bool] = np.max([int(wsi_dataframe['Gleason_primary'][row]) - 2, 0])
        dataframe['wsi_secondary_label'][id_bool] = np.max([int(wsi_dataframe['Gleason_secondary'][row]) - 2, 0])
    # assert(np.all(dataframe['wsi_index'] != -1))
    # assert(np.all(dataframe['wsi_primary_label'] != -1))
    # assert(np.all(dataframe['wsi_secondary_label'] != -1))
    return dataframe

def set_wsi_labels_cb(dataframe, wsi_dataframe):
    dataframe['wsi_index'] = -1
    dataframe["wsi_label"] = -1
    dataframe['wsi_contains_unlabeled'] = True
    for row in range(len(wsi_dataframe)):
        id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
        if np.any(id_bool) == False:
            continue
        dataframe['wsi_index'][id_bool] = row
        dataframe['wsi_label'][id_bool] = wsi_dataframe['class'][row].astype(int)
    assert(np.all(dataframe['wsi_index'] != -1))
    assert(np.all(dataframe['wsi_label'] != -1))
    return dataframe

# TODO: adapt to binary
def check_if_wsi_contains_unlabeled(dataframe, wsi_dataframe, dataset_type):
    if dataset_type == 'prostate_cancer':
        wsi_label_col = 'Gleason_primary'
        unlabeled_index = str(4)
    else:
        wsi_label_col = 'class'
        unlabeled_index = str(2)
    for row in range(len(wsi_dataframe)):
        id_bool = dataframe['wsi']==wsi_dataframe['slide_id'][row]
        if np.any(id_bool) == False:
            continue
        if wsi_dataframe[wsi_label_col][row] == '0':
            wsi_contains_unlabeled = False
        elif np.all(dataframe['class'][id_bool] != unlabeled_index):
            wsi_contains_unlabeled = False
        else:
            wsi_contains_unlabeled = True
        dataframe['wsi_contains_unlabeled'][id_bool] = wsi_contains_unlabeled
    return dataframe


def get_rows_of_visible_instances_pc(dataframe, wsi_dataframe, num_instance_samples):
    rows_of_visible_instance_labels = []
    for wsi_df_row in range(len(wsi_dataframe["Gleason_primary"])):
        if wsi_dataframe['Gleason_primary'][wsi_df_row] == wsi_dataframe['Gleason_secondary'][wsi_df_row] == 0:
            negative_bag = True
        else:
            negative_bag = False

        primary_gleason_grade_rows = []
        secondary_gleason_grade_rows = []
        for instance_df_row in range(len(dataframe["image_path"])):
            if wsi_dataframe['slide_id'][wsi_df_row] == dataframe["wsi"][instance_df_row]:
                if negative_bag:
                    rows_of_visible_instance_labels.append(instance_df_row)
                elif wsi_dataframe['Gleason_primary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
                    primary_gleason_grade_rows.append(instance_df_row)
                elif wsi_dataframe['Gleason_secondary'][wsi_df_row] - 2 == int(dataframe["class"][instance_df_row]):
                    secondary_gleason_grade_rows.append(instance_df_row)
        rows_of_visible_instance_labels += sample_or_complete_list(primary_gleason_grade_rows, num_instance_samples)
        rows_of_visible_instance_labels += sample_or_complete_list(secondary_gleason_grade_rows, num_instance_samples)
    return rows_of_visible_instance_labels

def get_rows_of_visible_instances_cb(dataframe, wsi_dataframe, num_instance_samples):
    rows_of_visible_instance_labels = []
    for wsi_df_row in range(len(wsi_dataframe)):
        if wsi_dataframe['class'][wsi_df_row] == 0:
            negative_bag = True
        else:
            negative_bag = False

        positive_rows = []
        for instance_df_row in range(len(dataframe["image_path"])):
            if wsi_dataframe['slide_id'][wsi_df_row] == dataframe['wsi'][instance_df_row]:
                if negative_bag:
                    rows_of_visible_instance_labels.append(instance_df_row)
                elif dataframe['class'][instance_df_row] == '1':
                    positive_rows.append(instance_df_row)

        rows_of_visible_instance_labels += sample_or_complete_list(positive_rows, num_instance_samples)
    return rows_of_visible_instance_labels


def sample_or_complete_list(list, num_samples):
    random.seed(42)
    if num_samples >= len(list):
        return list
    else:
        return random.sample(list, num_samples)