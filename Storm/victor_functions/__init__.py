"""
victor_functions contains specialised functions written for
the ACDS project The Day After Tomorrow written by team victor

"""
from .image_model import Seq2SeqAutoencoder, AutoencoderTrainer
from .image_loader import ImageDataset
from .data_loader import Storm_Dataset
from .model import ImageRegressionCNN  # ImageRegressionTrainer
from .training import ImageRegressionTrainer

__all__ = ['Storm_Dataset', 'ImageRegressionCNN', 'ImageRegressionTrainer',
           'Seq2SeqAutoencoder', 'train_model', 'ImageDataset',
           'AutoencoderTrainer']
