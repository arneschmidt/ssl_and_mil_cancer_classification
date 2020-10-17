import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Optional, Tuple

# TODO: add multiple instance learning setting
class DataGenerator():
    def __init__(self, data_config, batch_size):
        self.data_config = data_config
        self.batch_size = batch_size

    def generate_data(self):
        train_df, val_df, test_df = self.load_dataframes()

        train_generator = self.data_generator_from_dataframe(train_df, image_augmentation=True)
        validation_generator = self.data_generator_from_dataframe(val_df, image_augmentation=False)
        test_generator = self.data_generator_from_dataframe(test_df, image_augmentation=False)

        return train_generator, validation_generator, test_generator

    def data_generator_from_dataframe(self, dataframe: pd.DataFrame, image_augmentation=False):
        if image_augmentation:
            datagen = ImageDataGenerator(
                brightness_range=[0.9, 1.1],
                shear_range=0.2,
                zoom_range=0.0,
                horizontal_flip=True,
                vertical_flip=True)
        else:
            datagen = ImageDataGenerator()

        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=self.data_config["dir"],
            x_col="image_path",
            y_col="class",
            target_size=self.data_config["image_target_size"],
            batch_size=self.batch_size,
            class_mode="categorical")

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
            train_df = self.extract_sicap_df_info(train_df_raw)
            val_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
            val_df = self.extract_sicap_df_info(val_df_raw)
            test_df = val_df
        else:
            Exception("Please choose valid dataset name!")
        return train_df, val_df, test_df

    def extract_sicap_df_info(self, dataframe_raw):
        # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
        dataframe = pd.DataFrame()
        dataframe["image_path"] = dataframe_raw["image_name"]
        dataframe["class"] = np.argmax([dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]],
                                       axis=0).astype(str)
        return dataframe

