import os
import numpy as np

def save_dataframe_with_output(dataframe, predictions, features, output_dir, save_name):
    dataframe['predictions'] = np.argmax(predictions, axis=1)
    for feature_id in range(np.shape(features)[1]):
        feature_name = 'feature_' + str(feature_id)
        dataframe[feature_name] = features[:, feature_id]

    output_dir = os.path.join(output_dir, 'feature_predictions')
    os.makedirs(output_dir, exist_ok=True)
    save_path = output_dir + '/' + save_name + '.csv'
    print('Saving dataframe with output: ' + save_path)
    dataframe.to_excel(save_path, index=False)