import os
import numpy as np

def save_dataframe_with_output(dataframe, predictions, features, output_dir, save_name):
    dataframe['predictions'] = np.argmax(predictions, axis=1)
    for feature_id in range(np.shape(features)[1]):
        feature_name = 'feature_' + str(feature_id)
        dataframe[feature_name] = features[:, feature_id]

    output_dir = os.path.join(output_dir, 'feature_predictions')
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir + '/' + save_name + '.xlsx'
    print('Saving dataframe with output: ' + save_path)
    dataframe.to_excel(save_path, index=False)

def save_confusion_matrices(confusion_matrices, output_dir):
    output_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(output_dir, exist_ok=True)
    for name, matrix in confusion_matrices.items():
        save_path = output_dir + '/' + name + '.csv'
        np.savetxt(save_path, matrix, '%10i',  delimiter=',')
