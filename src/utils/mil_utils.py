import numpy as np


def combine_pseudo_labels_with_instance_labels(predictions, train_df, number_of_pseudo_labels_per_class):
    unlabeled_index = len(predictions[0]) # index of unlabeled class
    gt_labels = np.array(train_df['class'], dtype=int)
    predictions = pad_array(predictions, len(train_df))
    pseudo_labels = get_pseudo_labels(predictions, train_df, unlabeled_index, number_of_pseudo_labels_per_class)
    training_targets = np.where(gt_labels == unlabeled_index, pseudo_labels, gt_labels).astype(np.int) # choose pseudo lables only when gt unlabeled
    training_targets_one_hot_plus_unlabeled = get_one_hot(training_targets, unlabeled_index+1)
    training_targets_one_hot = training_targets_one_hot_plus_unlabeled[:, 0:unlabeled_index]
    training_targets_soft_and_one_hot = np.where((training_targets_one_hot_plus_unlabeled[:,unlabeled_index] == 1)[:,np.newaxis], predictions, training_targets_one_hot)
    return training_targets_soft_and_one_hot

def get_pseudo_labels(predictions, train_df, unlabeled_index, number_of_pseudo_labels_per_class):
    row = 0
    pseudo_labels = np.full(shape=len(predictions), fill_value=unlabeled_index)
    while True:
        wsi_name = train_df['wsi'][row]
        wsi_labels = train_df['wsi_labels'][row]
        wsi_df = train_df[train_df['wsi'].str.match(wsi_name)]
        end_row_wsi = row + len(wsi_df)

        if not wsi_labels[0] == wsi_labels[1] == 0: # means: positive bag
            for wsi_label in wsi_labels:
                sorted_indices_low_to_high = np.argsort(predictions[row:end_row_wsi,wsi_label], axis=0)
                top_indices = sorted_indices_low_to_high[::-1][:number_of_pseudo_labels_per_class]
                top_indices = top_indices + row
                pseudo_labels[top_indices] = wsi_label
        if end_row_wsi == len(train_df):
            break
        else:
            row = end_row_wsi
    return pseudo_labels

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def get_data_generator_with_targets(data_generator, targets):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        y_target = targets[indices]
        yield x, y_target

def get_data_generator_without_targets(data_generator):
    for x, _ in data_generator:
        yield x

def pad_array(array, length):
    padded_array = np.zeros(shape=(length, array.shape[1]))
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array