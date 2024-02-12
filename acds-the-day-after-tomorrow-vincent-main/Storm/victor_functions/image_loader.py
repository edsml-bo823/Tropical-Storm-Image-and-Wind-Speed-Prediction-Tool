import os
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import json


class ImageDataset():
    """
    Custom dataset for handling sequences of storm images for prediction.

    Args:
        root (str): Root directory containing storm image data.
        transform (bool, optional): Apply image transformations.
        Default is True.
        sequence_length (int, optional): Length of the image sequences.
        Default is 5.
        folder (str, optional): Subfolder containing storm data.
        Default is None.
    """

    def __init__(self, root, transform=True, sequence_length=5, stride=1,
                 folder=None):
        self.transform = transform
        self.root = root
        self.sequence_length = sequence_length
        self.folder = folder
        self.stride = stride
        self.data = self._get_all_storm_sequences()

    def _get_all_storm_sequences(self):
        """
        Retrieves all storm sequences from the specified directory.

        This method iterates over storm data folders to extract
        and compile storm sequences.
        If a specific folder is set in the class, it retrieves
        sequences from that folder only.
        Otherwise, it processes the first five folders in the
        root directory.

        Returns:
            list: A list of all storm sequences extracted
            from the folders.
        """
        all_sequences = []
        if self.folder:
            storm_path = os.path.join(self.root, self.folder)
            storm_sequences = self._get_storm_sequences(storm_path)
            for seq in storm_sequences:
                all_sequences.append(seq)
        else:
            for storm_folder in os.listdir(self.root)[:5]:
                storm_path = os.path.join(self.root, storm_folder)
                if os.path.isdir(storm_path):
                    storm_sequences = self._get_storm_sequences(storm_path)
                    for seq in storm_sequences:
                        all_sequences.append(seq)
        return all_sequences

    def _get_storm_sequences(self, storm_path):
        '''split storms into sequence'''
        paths = self._get_image_paths(storm_path)
        sequences = []
        for i in range(0, len(paths) - self.sequence_length, self.stride):
            sequence = paths[i:i + self.sequence_length]
            sequences.append(sequence)
        return sequences

    def _get_image_paths(self, storm_path, exts=(".jpg")):
        """
        Collects and sorts image paths from a
        given storm path based on their time feature.

        This method walks through the directory specified by `storm_path`,
        looking for images with specified extensions.
        It associates each image with its 'relative_time'
        feature from a corresponding JSON file, if available.

        Args:
            storm_path (str): The directory path to search for images.
            exts (tuple, optional): A tuple of file extensions to consider.
            Default is ('.jpg').

        Returns:
            list: Sorted list of image paths based on their time feature.
        """
        data = []
        for root, dirs, files in os.walk(storm_path):
            for file in files:
                if file.endswith(exts):
                    img_path = os.path.join(root, file)
                    num_path = img_path.removesuffix('.jpg')
                    try:
                        features_path = num_path + "_features.json"
                        if os.path.exists(features_path):
                            with open(features_path, 'r') as f:
                                features = json.load(f)
                                time_feature = features.get('relative_time')
                                if time_feature is not None:
                                    data.append((img_path, int(time_feature)))
                    except UnidentifiedImageError:
                        print('image error')
                        pass

        data.sort(key=lambda x: x[1])
        return [item[0] for item in data]

    def __getitem__(self, idx):
        """
        Retrieves a sequence of images and its corresponding
        target image for a given index.

        This method selects a sequence of image paths based on
        the index, loads the images,
        and optionally applies transformations. The last image
        in the sequence is treated as
        the target image. All images are converted to grayscale.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple: A tuple containing a stack of images (as tensors)
            and the target image tensor.
        """
        sequence_paths = self.data[idx]
        images = []
        for img_path in sequence_paths[:-1]:
            img = Image.open(img_path).convert('L')
            if self.transform:
                transform = transforms.Compose([
                    transforms.Resize((336, 336)),
                    transforms.ToTensor()
                ])
                img = transform(img)
            images.append(img)

        target_img_path = sequence_paths[-1]
        target_img = Image.open(target_img_path).convert('L')
        if self.transform:
            target_img = transform(target_img)

        return torch.stack(images), target_img

    def __len__(self):
        return len(self.data)

    def __str__(self):
        class_string = self.__class__.__name__
        class_string += f"\n\tlen : {self.__len__()}"
        for key, value in self.__dict__.items():
            class_string += f"\n\t{key} : {value}"
        return class_string
