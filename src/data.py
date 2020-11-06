import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from random import sample
from utils.sicapv2_utils import extract_sicap_df_info

# TODO: add multiple instance learning setting
class DataGenerator():
    def __init__(self, data_config, batch_size):
        self.data_config = data_config
        self.num_classes = data_config['num_classes']
        self.batch_size = batch_size
        train_df, val_df, test_df = self.load_dataframes()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        if data_config['supervision'] == 'mil':
            self.train_generator_strong_aug = self.data_generator_from_dataframe(train_df, image_augmentation='strong')
            self.train_generator_weak_aug = self.data_generator_from_dataframe(train_df, image_augmentation='weak',
                                                                               shuffle='False', class_mode=None)
            self.num_training_samples = self.train_generator_weak_aug.n
        else:
            self.train_generator = self.data_generator_from_dataframe(train_df, image_augmentation='weak')
            self.num_training_samples = self.train_generator.n
        self.validation_generator = self.data_generator_from_dataframe(val_df)
        self.test_generator = self.data_generator_from_dataframe(test_df)

    def data_generator_from_dataframe(self, dataframe: pd.DataFrame, image_augmentation='None', shuffle='True',
                                      class_mode='raw'):
        if image_augmentation == 'weak':
            datagen = ImageDataGenerator(
                brightness_range=[0.9, 1.1],
                shear_range=0.2,
                zoom_range=0.0,
                horizontal_flip=True,
                vertical_flip=True)
        elif image_augmentation == 'strong':
            datagen = ImageDataGenerator(
                brightness_range=[0.6, 1.5],
                rotation_range=360,
                shear_range=0.5,
                zoom_range=0.0,
                horizontal_flip=True,
                vertical_flip=True)
        else:
            datagen = ImageDataGenerator()

        dataframe = self.convert_to_one_hot(dataframe)
        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=self.data_config["dir"],
            x_col="image_path",
            y_col="class_one_hot",
            target_size=self.data_config["image_target_size"],
            batch_size=self.batch_size,
            shuffle=shuffle,
            class_mode=class_mode)

        return generator

    def load_dataframes(self):
        if self.data_config["dataset_name"] == "breast_hist_images":
            train_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train.txt"))
            train_df["class"] = train_df["image_path"].str.extract("class(\d+)").astype(str)
            val_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val.txt"))
            val_df["class"] = val_df["image_path"].str.extract("class(\d+)").astype(str)
            test_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test.txt"))
            test_df["class"] = test_df["image_path"].str.extract("class(\d+)").astype(str)
        elif self.data_config["dataset_name"] == "sicapv2":
            train_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Train.xlsx"))
            train_df = extract_sicap_df_info(train_df_raw, self.data_config)
            val_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
            val_df = extract_sicap_df_info(val_df_raw, self.data_config)
            test_df = val_df
        else:
            raise Exception("Please choose valid dataset name!")
        return train_df, val_df, test_df

    def convert_to_one_hot(self, dataframe):
        labels = dataframe['class']
        if self.data_config['supervision'] == 'full':
            class_one_hot = to_categorical(labels, num_classes=self.num_classes)
        else:
            class_one_hot = to_categorical(labels, num_classes=self.num_classes+1)
            class_one_hot = class_one_hot[:, 0:self.num_classes]

        dataframe["class_one_hot"] = np.NaN
        dataframe["class_one_hot"] = dataframe["class_one_hot"].astype(object)
        dataframe['class_one_hot'] = dataframe['class_one_hot'].astype(object)
        for i in range(len(class_one_hot)):
            dataframe['class_one_hot'].iloc[i] = class_one_hot[i]
        return dataframe