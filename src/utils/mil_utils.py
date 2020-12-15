import numpy as np


def combine_pseudo_labels_with_instance_labels(predictions, train_df, number_of_pseudo_labels_per_class, label_weights):
    unlabeled_index = len(predictions[0]) # index of unlabeled class
    gt_labels = np.array(train_df['class'], dtype=int)
    # predictions = pad_array(predictions, len(train_df))
    pseudo_labels = get_pseudo_labels(predictions, train_df, unlabeled_index, number_of_pseudo_labels_per_class)
    training_targets = np.where(gt_labels == unlabeled_index, pseudo_labels, gt_labels).astype(np.int) # choose pseudo lables only when gt unlabeled

    sample_weights = _calculate_sample_weights(gt_labels, training_targets, label_weights, unlabeled_index)
    training_targets_soft_and_one_hot = convert_to_one_hot_and_soft_labels(training_targets, predictions, unlabeled_index)

    return training_targets_soft_and_one_hot, sample_weights

def get_pseudo_labels(predictions, train_df, unlabeled_index, number_of_pseudo_labels_per_class):
    row = 0
    pseudo_labels = np.full(shape=len(predictions), fill_value=unlabeled_index)
    while True:
        # select
        wsi_name = train_df['wsi'].iloc[row]
        wsi_labels = [train_df['wsi_primary_label'][row], train_df['wsi_secondary_label'][row]]
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
        elif end_row_wsi >= len(train_df):
            print('row: ' + str(row))
            print('end_row_wsi: ' + str(end_row_wsi))
            print('train_df[].iloc[row-1]:')
            print(train_df['wsi'].iloc[row - 1])
            raise Exception('Error in pseudo labeling with dataframes')
        else:
            row = end_row_wsi
    return pseudo_labels

def convert_to_one_hot_and_soft_labels(training_targets, predictions, unlabeled_index):
    training_targets_one_hot_plus_unlabeled = get_one_hot(training_targets, unlabeled_index+1)
    training_targets_one_hot = training_targets_one_hot_plus_unlabeled[:, 0:unlabeled_index]
    training_targets_soft_and_one_hot = np.where((training_targets_one_hot_plus_unlabeled[:,unlabeled_index] == 1)[:,np.newaxis], predictions, training_targets_one_hot)

    return training_targets_soft_and_one_hot

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def get_data_generator_with_targets(data_generator, targets, sample_weights):
    for x, y in data_generator:
        indices = y.astype(np.int).tolist()
        y_target = targets[indices]
        y_sample_weight = sample_weights[indices]
        yield x, y_target, y_sample_weight

def get_data_generator_without_targets(data_generator):
    for x, _ in data_generator:
        yield x

def pad_array(array, length):
    padded_array = np.zeros(shape=(length, array.shape[1]))
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array

def _calculate_sample_weights(gt_labels, pseudo_labels, label_weights, unlabeled_index):
    number_targets = len(gt_labels)
    positive_gt_labels_weight_array = np.full(number_targets, fill_value=label_weights['positive_gt_labels'])
    pseudo_labels_weight_array = np.full(number_targets, fill_value=label_weights['pseudo_labels'])
    soft_label_weight_array = np.full(number_targets, fill_value=label_weights['soft_labels'])
    negative_gt_labels_weight_array = np.full(number_targets, fill_value=label_weights['negative_gt_labels'])

    sample_weights = np.full(number_targets, fill_value=0)
    sample_weights = np.where(gt_labels == 0, negative_gt_labels_weight_array, sample_weights)
    sample_weights = np.where(np.logical_and(pseudo_labels != unlabeled_index, gt_labels != 0), pseudo_labels_weight_array, sample_weights)
    sample_weights = np.where(np.logical_and((gt_labels != 0), (gt_labels != unlabeled_index)), positive_gt_labels_weight_array, sample_weights)
    sample_weights = np.where(np.logical_and(pseudo_labels == unlabeled_index, gt_labels == unlabeled_index), soft_label_weight_array, sample_weights)
    assert np.all(sample_weights != 0)

    return sample_weights


