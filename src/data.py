import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Optional, Tuple


def generate_data(dataset_name: str, data_dir: str, split_dir: str,
                  image_target_size: Tuple, batch_size: int):
    train_df, val_df, test_df = load_dataframes(dataset_name, split_dir)

    train_generator = data_generator_from_dataframe(data_dir, train_df, batch_size, image_target_size, image_augmentation=True)
    validation_generator = data_generator_from_dataframe(data_dir, val_df, batch_size, image_target_size, image_augmentation=False)
    test_generator = data_generator_from_dataframe(data_dir, test_df, batch_size, image_target_size, image_augmentation=False)

    return train_generator, validation_generator, test_generator

def data_generator_from_dataframe(data_dir: str, dataframe: pd.DataFrame, batch_size:int, image_target_size:int, image_augmentation=False):
    if image_augmentation:
        datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
    else:
        datagen = ImageDataGenerator()

    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=data_dir,
        x_col="image_path",
        y_col="class",
        target_size=image_target_size,
        batch_size=batch_size,
        class_mode="categorical")

    return generator

def load_dataframes(dataset_name: str, filenames_path: str):
    if dataset_name == "breast_hist_images":
        train_df = pd.read_csv(os.path.join(filenames_path, "train.txt"))
        train_df["class"] = train_df["image_path"].str.extract("class(\d+)").astype(str)
        val_df = pd.read_csv(os.path.join(filenames_path, "val.txt"))
        val_df["class"] = val_df["image_path"].str.extract("class(\d+)").astype(str)
        test_df = pd.read_csv(os.path.join(filenames_path, "test.txt"))
        test_df["class"] = test_df["image_path"].str.extract("class(\d+)").astype(str)
    elif dataset_name == "sicapv2":
        train_df_raw = pd.read_excel(os.path.join(filenames_path, "Train.xlsx"))
        train_df = extract_sicap_df_info(train_df_raw)
        val_df_raw = pd.read_excel(os.path.join(filenames_path, "Test.xlsx"))
        val_df = extract_sicap_df_info(val_df_raw)
        test_df = val_df
    else:
        Exception("Please choose valid dataset name!")
    return train_df, val_df, test_df

def extract_sicap_df_info(dataframe_raw):
    # Notice: class 0 = NC, class 1 = G3, class 2 = G4, class 3 = G5
    dataframe = pd.DataFrame()
    dataframe["image_path"] = dataframe_raw["image_name"]
    dataframe["class"] = np.argmax([dataframe_raw["NC"], dataframe_raw["G3"], dataframe_raw["G4"], dataframe_raw["G5"]],
                                   axis=0).astype(str)
    return dataframe

