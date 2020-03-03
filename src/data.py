import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def data_generator_from_dataframe(data_dir, dataframe, batch_size, image_augmentation=False):
    dataframe["class"] = dataframe["image_path"].str.extract("class(\d+)").astype(str)
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
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical")

    return generator


def generate_data(data_dir, filenames_path, batch_size):
    train_df = pd.read_csv(os.path.join(filenames_path, "train.txt"))
    val_df = pd.read_csv(os.path.join(filenames_path, "val.txt"))
    test_df = pd.read_csv(os.path.join(filenames_path, "test.txt"))

    train_generator = data_generator_from_dataframe(data_dir, train_df, batch_size,  image_augmentation=True)
    validation_generator = data_generator_from_dataframe(data_dir, val_df, batch_size,  image_augmentation=False)
    test_generator = data_generator_from_dataframe(data_dir, test_df, batch_size,  image_augmentation=False)

    return train_generator, validation_generator, test_generator
