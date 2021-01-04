import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_utils import extract_df_info

# TODO: add multiple instance learning setting
class DataGenerator():
    def __init__(self, data_config, batch_size, mode):
        self.data_config = data_config
        self.num_classes = data_config['num_classes']
        self.batch_size = batch_size
        self.mode = mode
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.wsi_df = None
        self._create_data_generators(mode, data_config)

    def _create_data_generators(self, mode, data_config):
        if mode == 'train' or mode == 'predict_features' or mode == 'predict':
            self.load_dataframes(split='train')
            if data_config['supervision'] == 'mil':
                self.train_generator_strong_aug = self.data_generator_from_dataframe(self.train_df, image_augmentation='strong',
                                                                                     shuffle=True, target_mode='index')
                self.train_df_weak_aug = self.train_df[self.train_df['wsi_contains_unlabeled']]
                self.train_generator_weak_aug = self.data_generator_from_dataframe(self.train_df_weak_aug, image_augmentation='weak',
                                                                                   shuffle=False, target_mode='None')
                self.num_training_samples = self.train_generator_weak_aug.n
            else:
                self.train_generator = self.data_generator_from_dataframe(self.train_df, image_augmentation='strong', shuffle=True)
                self.num_training_samples = self.train_generator.n
            self.validation_generator = self.data_generator_from_dataframe(self.val_df)
        elif mode =='test':
            self.load_dataframes(split='test')
            self.test_generator = self.data_generator_from_dataframe(self.test_df)
            self.num_training_samples = self.test_generator.n
        else:
            raise Exception('Choose valid model mode')

    def data_generator_from_dataframe(self, dataframe: pd.DataFrame, image_augmentation='None', shuffle=False,
                                      target_mode='class'):
        if image_augmentation == 'weak':
            datagen = ImageDataGenerator(
                brightness_range=self.data_config["weak_augment_brightness_range"],
                channel_shift_range=self.data_config["weak_augment_channel_shift"],
                rotation_range=360,
                fill_mode='reflect',
                horizontal_flip=True,
                vertical_flip=True)
        elif image_augmentation == 'strong':
            datagen = ImageDataGenerator(
                brightness_range=self.data_config["strong_augment_brightness_range"],
                channel_shift_range=self.data_config["strong_augment_channel_shift"],
                rotation_range=360,
                fill_mode='reflect',
                horizontal_flip=True,
                vertical_flip=True)
        else:
            datagen = ImageDataGenerator()

        if target_mode == 'class':
            y_col = 'class'
            class_mode = 'categorical'
            classes = [str(i) for i in range(self.num_classes)]
        elif target_mode == 'index':
            y_col = 'index'
            class_mode = 'raw'
            classes = None
        else:
            y_col = 'index'
            class_mode = None
            classes = None

        dataframe['index'] = dataframe.index

        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=self.data_config["dir"],
            x_col="image_path",
            y_col=y_col,
            target_size=self.data_config["image_target_size"],
            batch_size=self.batch_size,
            shuffle=shuffle,
            classes=classes,
            class_mode=class_mode,
            # save_to_dir=self.data_config['artifact_dir'] + '/' + image_augmentation,
            # save_format='jpeg'
            )

        return generator

    def load_dataframes(self, split):
        if self.data_config["dataset_name"] == "breast_hist_images":
            if split == 'train':
                train_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train.txt"))
                train_df["class"] = train_df["image_path"].str.extract("class(\d+)").astype(str)
                self.train_df = train_df
                val_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val.txt"))
                val_df["class"] = val_df["image_path"].str.extract("class(\d+)").astype(str)
                self.val_df = val_df
            elif split == 'test':
                test_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test.txt"))
                test_df["class"] = test_df["image_path"].str.extract("class(\d+)").astype(str)
                self.test_df = test_df
        elif self.data_config["dataset_name"] == "sicapv2":
            self.wsi_df = pd.read_excel(os.path.join(self.data_config["dir"], "wsi_labels.xlsx"))
            if split == 'train':
                train_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Train.xlsx"))
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train')
                val_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
            elif split == 'test':
                test_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test')
        elif self.data_config["dataset_name"] == "panda":
            wsi_df = pd.read_csv(os.path.join(self.data_config["dir"], "wsi_labels.csv"))
            wsi_df['Gleason_primary'] = wsi_df['gleason_score'].str.split('+').str[0]
            wsi_df['Gleason_secondary'] = wsi_df['gleason_score'].str.split('+').str[1]
            wsi_df.rename(columns={"image_id": "slide_id"}, inplace=True)
            self.wsi_df = wsi_df
            if split == 'train':
                train_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train_patches.csv"))
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train')
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val_patches.csv"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
            elif split == 'test':
                test_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test_patches.csv"))
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test')
        else:
            raise Exception("Please choose valid dataset name!")