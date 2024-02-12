import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from operator import itemgetter
import math


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return True


class Storm_Dataset(Dataset):
    """
    A custom dataset class for loading image data along with their associated
    labels and features.

    This class is designed to load data from a specified directory, handling
    images and their associated
    label and feature files. It splits the data into training and test sets
    and supports basic transformations.

    Attributes:
        root (str): The root directory from which the data is loaded.
        data (list): List of tuples containing paths to images and their
        associated labels and features.
        train_data (list): List of training data samples.
        test_data (list): List of test data samples.

    Args:
        root (str): The directory path from where to load the data.
        test_size (float, optional): Proportion of the dataset to include in
        the test split. Defaults to 0.2.
        seed (int, optional): Random seed for splitting the dataset.
        Defaults to 42.
    """

    def __init__(self, root,
                 test_size=0.2, seed=42, split_method='random', split='none'):
        self.root = root
        all_data = self._get_image_paths()
        self.data = all_data
        # Splitting the dataset into training and testing
        # Also allows for no split for data analysis
        if split_method == 'random':
            train, test = train_test_split(
                all_data, test_size=test_size, random_state=seed)
            if split == 'train':
                self.data = train
            elif split == 'test':
                self.data = test
        elif split_method == 'time':
            split_index = math.floor(self.__len__() * 0.7)
            train = self.data[:split_index]
            test = self.data[split_index:]
            if split == 'train':
                self.data = train
            elif split == 'test':
                self.data = test

    def _get_image_paths(self, exts=(".jpg")):
        """
        Private method to scan the directory and gather image paths along with
        their label and feature data.

        This method walks through the directory, finds images with specified
        extensions, and attempts to locate corresponding label and feature
        files. It handles basic exceptions related to image loading.

        Args:
            exts (tuple of str, optional): File extensions to consider for
            images. Defaults to (".jpg").

        Returns:
            list: A list of tuples, each containing paths to an image, its
            label, and features.
        """
        data = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(exts):
                    path = os.path.join(root, file)
                    num_path = path.removesuffix('.jpg')

                    try:
                        label_path = num_path + "_label.json"
                        features_path = num_path + "_features.json"
                        label, storm_id, time, ocean = None, None, None, None
                        if os.path.exists(label_path):
                            with open(label_path, 'r') as f:
                                label_data = json.load(f)
                                label = int(label_data['wind_speed'])

                        if os.path.exists(features_path):
                            with open(features_path, 'r') as f:
                                features_data = json.load(f)
                                storm_id = features_data['storm_id']
                                time = int(features_data['relative_time'])
                                ocean = int(features_data['ocean'])

                        data.append((path, label, storm_id, time, ocean))
                    except UnidentifiedImageError:
                        print('image error')
                        pass
        data.sort(key=itemgetter(3))
        return data

    def __getitem__(self, idx):
        """
        Returns a single data sample at the specified index.

        This method reads the image from the disk, applies transformations if
        required, and returns it along with its label and features.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image, label, and
            features of the sample.
        """
        img_path, label,  storm_id, time, ocean = self.data[idx]
        img = Image.open(img_path)
        img = img.convert('L')
        img = ToTensor()(img)
        return img, label, storm_id, time, ocean

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __str__(self):
        """
        Returns a string representation of the dataset object.

        Provides a summary of the dataset including its length and other
        attributes except the raw data.

        Returns:
            str: A string summarizing the dataset.
        """
        class_string = self.__class__.__name__
        class_string += f"\n\tlen : {self.__len__()}"
        for key, value in self.__dict__.items():
            if key != "data":
                class_string += f"\n\t{key} : {value}"
        return class_string
