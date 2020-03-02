import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def generate_data(data_dir, filenames_path, batch_size):
    train_df = pd.read_csv(filenames_path)
    train_df["class"] = train_df["image_path"].str.extract("class(\d+)").astype(str)
    train_df = train_df.dropna(thresh=1)
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=data_dir,
        x_col="image_path",
        y_col="class",
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode="categorical")

    return train_generator
