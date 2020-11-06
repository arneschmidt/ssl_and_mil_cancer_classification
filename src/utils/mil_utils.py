import numpy as np


def combine_pseudo_labels_with_instance_labels(predictions, data_gen):
    unlabeled_index = len(predictions[0])
    gt_labels = np.array(data_gen.train_df['class'])
    pseudo_labels = get_pseudo_labels(predictions, data_gen, unlabeled_index)
    training_targets = np.where(gt_labels == unlabeled_index, pseudo_labels, gt_labels).astype(np.int) # choose pseudo lables only when gt unlabeled
    training_targets_one_hot_plus_unlabeled = get_one_hot(training_targets, unlabeled_index+1)
    training_targets_one_hot = training_targets_one_hot_plus_unlabeled[:, 0:unlabeled_index]
    training_targets_soft_and_one_hot = np.where((training_targets_one_hot_plus_unlabeled[:,unlabeled_index] == 1)[:,np.newaxis], predictions, training_targets_one_hot)
    return training_targets_soft_and_one_hot

def get_pseudo_labels(predictions, data_gen, unlabeled_index):
    train_dataframe = data_gen.train_df
    row = 0
    number_of_pseudo_labels_per_class = 5
    pseudo_labels = np.full(shape=len(predictions), fill_value=unlabeled_index)
    while True:
        wsi_name = train_dataframe['wsi'][row]
        wsi_labels = train_dataframe['wsi_labels'][row]
        wsi_df = train_dataframe[train_dataframe['wsi'].str.match(wsi_name)]
        end_row_wsi = row + len(wsi_df)

        if len(wsi_labels) >= 1: # means: positive bag
            for wsi_label in wsi_labels[1:]:
                indices = np.argsort(predictions[row:end_row_wsi,wsi_label], axis=0)[0:number_of_pseudo_labels_per_class]
                indices = indices + row
                pseudo_labels[indices] = wsi_label
        if end_row_wsi == len(train_dataframe):
            break
        else:
            row = end_row_wsi
    return pseudo_labels

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])