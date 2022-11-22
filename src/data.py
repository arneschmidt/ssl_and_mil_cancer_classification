import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_utils import extract_df_info
from typing import Dict, Optional, Tuple
from keras_preprocessing import image

class DataGenerator():
    """
    Object to obtain the patches and labels.
    """
    def __init__(self, config: Dict):
        """
        Initialize data generator object
        :param config: dict containing config
        """
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.wsi_df = None
        self.num_training_samples = 0
        self._create_data_generators()

    def _create_data_generators(self):
        """
        Create data generators for supervised or semi-supervised multiple instance learning.
        The number of patch labels specified by 'positive_instance_labels_per_bag' is automatically obtained by
        randomly masking patch labels.

        :return: Keras image data generator providing the patches and (if available) labels.
        """
        mode = self.model_config["mode"]
        data_config = self.data_config
        if mode == 'train':
            self.load_dataframes(split='train')
            # Init setting of semi-supervised MIL training
            if data_config['supervision'] in ['mil', 'ssl']:
                self.train_generator_strong_aug, self.train_df = self.data_generator_from_dataframe(self.train_df, image_augmentation='strong',
                                                                                     shuffle=True, target_mode='index')
                self.train_generator_weak_aug, self.train_df_weak_aug = self.data_generator_from_dataframe(self.train_df_weak_aug, image_augmentation='weak',
                                                                                   shuffle=False, target_mode='None')
                self.num_training_samples = self.train_generator_weak_aug.n
            # Init supervised training setting
            else:
                self.train_df = self.train_df[self.train_df['class'] != self.data_config['num_classes']].reset_index()
                self.train_generator_strong_aug, self.train_df = self.data_generator_from_dataframe(self.train_df, image_augmentation='strong',
                                                                                     shuffle=True, target_mode='index')
                self.train_generator_weak_aug = self.train_generator_strong_aug
                self.num_training_samples = self.train_generator_strong_aug.n
            self.validation_generator, self.val_df = self.data_generator_from_dataframe(self.val_df, target_mode='raw')
        elif mode =='test' or mode == 'predict':
            self.load_dataframes(split='test')
            self.validation_generator, self.val_df = self.data_generator_from_dataframe(self.val_df, target_mode='raw')
            self.test_generator, self.test_df = self.data_generator_from_dataframe(self.test_df, target_mode='raw')
            self.num_training_samples = self.test_generator.n # just formally necessary for model initialization
        elif mode == 'predict_features':
            self.load_dataframes(split='train')
            self.train_generator_weak_aug, self.train_df = self.data_generator_from_dataframe(self.train_df,
                                                                               image_augmentation='weak',
                                                                               shuffle=False, target_mode='None')
            self.validation_generator, self.val_df = self.data_generator_from_dataframe(self.val_df, target_mode='raw')
            self.load_dataframes(split='test')
            self.test_generator, self.test_df = self.data_generator_from_dataframe(self.test_df, target_mode='raw')
        else:
            raise Exception('Choose valid model mode')

    def data_generator_from_dataframe(self, dataframe: pd.DataFrame, image_augmentation: str = 'None',
                                      shuffle: bool = False, target_mode: str = 'class'):
        """
        Wrapper around 'flow_from_dataframe'-method. Uses loaded dataframes to load images and labels.

        :param dataframe: dataframe containing patch paths and labels
        :param image_augmentation: 'strong','weak' or 'None' indicating the level of augmentation
        :param shuffle: bool to shuffle the data after each epoch
        :param target_mode: 'class': loads patch classes, 'index': loads indices instead, or 'None' only loads images
        :return: data generator loading patches and labels (or indices)
        """
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
            classes = [str(i) for i in range(self.data_config["num_classes"])]
        elif target_mode == 'index':
            y_col = 'index'
            class_mode = 'raw'
            classes = None
        else:
            y_col = 'index'
            class_mode = None
            classes = None

        def validate_filename(filename, white_list_formats):
            return (filename.lower().endswith(white_list_formats) and
                    os.path.isfile(filename))
        filepaths = dataframe["image_path"].map(
            lambda fname: os.path.join(self.data_config["dir"], fname)
        )
        mask = filepaths.apply(validate_filename, args=(('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'),))
        dataframe = dataframe[mask]
        n_invalid = (~mask).sum()
        if n_invalid:
            print('Warning. Found ' + str(n_invalid) + 'invalid filnames')
        dataframe = dataframe.reset_index(drop=True)
        dataframe['index'] = dataframe.index

        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=self.data_config["dir"],
            x_col="image_path",
            y_col=y_col,
            target_size=self.data_config["image_target_size"],
            batch_size=self.model_config["batch_size"],
            shuffle=shuffle,
            classes=classes,
            class_mode=class_mode,
            # save_to_dir=self.data_config['artifact_dir'] + '/' + image_augmentation,
            # save_format='jpeg'
            )
        if len(dataframe) > len(generator):
            dataframe = dataframe[dataframe['image_path'].isin(generator.filenames)]
            print('Attention! Not all images in Dataframe were loaded.')

        return generator, dataframe

    def load_dataframes(self, split):
        """
        Load tables containing the patch paths and potentially classes.
        Loaded dataframes are stored as member variables.
        :param split: 'train', 'val' or 'test'
        """
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
        elif self.data_config["dataset_name"] == "camelyon16":
            wsi_df = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "wsi_labels.csv")).drop_duplicates().reset_index()
            wsi_df['class'] = wsi_df['P'].astype(int)
            wsi_df.rename(columns={"slide": "slide_id"}, inplace=True)
            self.wsi_df = wsi_df
            if split == 'train':
                train_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train.csv"))
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train')
                self.train_df_weak_aug = self.train_df[self.train_df['wsi_contains_unlabeled']]
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val.csv"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
            elif split == 'test':
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val.csv"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
                test_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test.csv"))
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test')
        elif self.data_config["dataset_name"] == "sicapv2":
            self.wsi_df = pd.read_excel(os.path.join(self.data_config["dir"], "wsi_labels.xlsx"))
            if split == 'train':
                train_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Train.xlsx"))
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train')
                self.train_df_weak_aug = self.train_df[self.train_df['wsi_contains_unlabeled']]
                val_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
            elif split == 'test':
                val_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
                test_df_raw = pd.read_excel(os.path.join(self.data_config["data_split_dir"], "Test.xlsx"))
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test')
        elif self.data_config["dataset_name"] == "panda":
            wsi_df = pd.read_csv(os.path.join(self.data_config["dir"], "wsi_labels.csv"))
            wsi_df['Gleason_primary'] = wsi_df['gleason_score'].str.split('+').str[0].astype(int)
            wsi_df['Gleason_secondary'] = wsi_df['gleason_score'].str.split('+').str[1].astype(int)
            wsi_df.rename(columns={"image_id": "slide_id"}, inplace=True)
            self.wsi_df = wsi_df
            if split == 'train':
                train_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train_patches.csv"))
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train')
                self.train_df_weak_aug = self.train_df[self.train_df['wsi_contains_unlabeled']]
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val_patches.csv"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
            elif split == 'test':
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val_patches.csv"))
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val')
                test_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test_patches.csv"))
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test')
        elif self.data_config["dataset_name"] == "prostate_ugr":
            wsi_df = pd.read_csv(os.path.join(self.data_config["dir"], "wsi_labels.csv"))
            wsi_df['Gleason_primary'] = wsi_df['gleason_score'].str.split('+').str[0].astype(int)
            wsi_df['Gleason_secondary'] = wsi_df['gleason_score'].str.split('+').str[1].astype(int)
            wsi_df.rename(columns={"image_id": "slide_id"}, inplace=True)
            self.wsi_df = wsi_df
            if split == 'train':
                train_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "train_patches.csv"))
                train_df_raw['image_name'] = train_df_raw['image_name'] + '.jpg'
                self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.data_config, split='train', wsi_delimiter='_row')
                self.train_df_weak_aug = self.train_df[self.train_df['wsi_contains_unlabeled']]
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val_patches.csv"))
                val_df_raw['image_name'] = val_df_raw['image_name'] + '.jpg'
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val', wsi_delimiter='_row')
            elif split == 'test':
                val_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "val_patches.csv"))
                val_df_raw['image_name'] = val_df_raw['image_name'] + '.jpg'
                self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.data_config, split='val', wsi_delimiter='_row')
                test_df_raw = pd.read_csv(os.path.join(self.data_config["data_split_dir"], "test_patches.csv"))
                test_df_raw['image_name'] = test_df_raw['image_name'] + '.jpg'
                self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.data_config, split='test', wsi_delimiter='_row')
        else:
            raise Exception("Please choose valid dataset name!")

    def get_train_data_statistics(self):
        """
        Calculate the number of labeled patches, classes and WSIs for statistics.
        :return: dict of statistics
        """
        train_df = self.train_df
        wsi_df = self.wsi_df
        wsi_names = np.unique(np.array(train_df['wsi']))
        out_dict = {}
        out_dict['number_of_wsis'] = len(wsi_names)
        out_dict['number_of_patches'] = len(train_df)
        if self.data_config["dataset_type"] == "prostate_cancer":
            out_dict['number_of_negative_patch_labels'] = np.sum(train_df['class'] == '0')
            out_dict['number_of_positive_patch_labels'] = np.sum(train_df['class'] == '1')\
                                                          + np.sum(train_df['class'] == '2') \
                                                          + np.sum(train_df['class'] == '3')
            out_dict['number_of_unlabeled_patches'] = np.sum(train_df['class'] == '4')
        else:
            out_dict['number_of_negative_patch_labels'] = np.sum(train_df['class'] == '0')
            out_dict['number_of_positive_patch_labels'] = np.sum(train_df['class'] == '1')
            out_dict['number_of_unlabeled_patches'] = np.sum(train_df['class'] == '2')

        return out_dict

