from livelossplot import PlotLosses
import torch
import torch.nn as nn
import numpy as np
import random

from .model import ImageRegressionCNN # noqa


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take
    out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


class ImageRegressionTrainer:
    """
    A trainer class for the ImageRegressionCNN model.

    This class encapsulates the training and validation process
    for an image regression CNN.
    It supports both training from scratch and transfer learning.

    Attributes:
        train_loader (DataLoader):
        DataLoader for the training dataset.
        val_loader (DataLoader):
        DataLoader for the validation dataset.
        device (str): The device (CPU or GPU) used for training.
        Defaults to 'cpu'.
        seed (int): Random seed for reproducibility. Defaults to 42.
        n_epochs (int): Number of epochs for training. Defaults to 100.
        model (ImageRegressionCNN): The ImageRegressionCNN model
        instance.
        criterion (nn.MSELoss): The loss function used for training
        (Mean Squared Error Loss).
        optimizer (torch.optim.Adam): The optimizer for training.
        pt_file (str, optional): Path to a pre-trained model file for
        transfer learning.

    Methods:
        train(): Runs a single training epoch.
        Returns the average training loss for the epoch.
        validate(): Runs validation on the validation dataset.
        Returns the average validation loss.
        train_model(transfer_learning=False): Orchestrates the
        training process over
        multiple epochs, including optional transfer learning. If
        `transfer_learning` is
        True, loads weights from `pt_file`. Returns the trained model.

    Raises:
        Exception: If `transfer_learning` is True and no `pt_file` is
        provided, an exception is raised.
    """
    def __init__(self, model, train_loader, val_loader, device='cpu',
                 seed=42, n_epochs=100, pt_file=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.seed = seed
        self.n_epochs = n_epochs
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          weight_decay=1.5e-5, lr=2e-3)
        self.pt_file = pt_file

    def train(self):
        """
        Trains the model for one epoch over the training dataset.

        Iterates over batches from `train_loader`, computes loss, and updates
        model weights.

        Returns:
            torch.Tensor: Average loss over the training dataset.

        Note:
            - Input format: single-channel, 256x256 pixels.
            - Operates in training mode.
        """
        self.model.train()
        train_loss = 0
        for X, y, t in self.train_loader:

            X, y = X.to(self.device), y.to(self.device).float()
            self.optimizer.zero_grad()
            output = self.model(X.view(-1, 1, 256, 256)).squeeze()
            loss = self.criterion(output, y)
            loss.backward()
            train_loss += loss*X.size(0)
            self.optimizer.step()
        return train_loss/len(self.train_loader.dataset)

    def validate(self):
        """
        Evaluates the model on the validation dataset.

        Iterates over batches from `val_loader`, computes loss without
        gradient calculation.

        Returns:
            torch.Tensor: Average loss over the validation dataset.

        Note:
            - Input format: single-channel, 256x256 pixels.
            - No gradient computation during evaluation.
        """
        self.model.eval()
        validation_loss = 0
        for X, y, t in self.val_loader:

            with torch.no_grad():
                X, y = X.to(self.device), y.to(self.device).float()
                output = self.model(X.view(-1, 1, 256, 256)).squeeze()
                loss = self.criterion(output, y)
                validation_loss += loss*X.size(0)

        return validation_loss/len(self.val_loader.dataset)

    def train_model(self, transfer_learning=False):
        """
        Trains the model, optionally using transfer learning.

        If `transfer_learning` is True, loads weights from a pre-trained model
        specified in `self.pt_file`. Iterates over epochs, trains and validates
        the model, and logs losses using `PlotLosses`.

        Args:
            transfer_learning (bool): If True, use transfer learning.

        Raises:
            Exception: If `transfer_learning` is True but no pre-trained file
            is provided in `self.pt_file`.

        Returns:
            The trained model.

        """
        if transfer_learning:
            if self.pt_file is not None:
                if self.device == 'cpu':
                    self.model.load_state_dict(torch.load(self.pt_file, map_location=torch.device('cpu')))  # noqa
                else:
                    self.model.load_state_dict(torch.load(self.pt_file))
            else:
                raise Exception('No pre-trained file provided')
        liveloss = PlotLosses()
        for epoch in range(self.n_epochs):
            logs = {}
            train_loss = self.train()
            logs['' + 'log loss'] = train_loss.item()
            validation_loss = self.validate()
            logs['val_' + 'log loss'] = validation_loss.item()

            liveloss.update(logs)
            liveloss.draw()
        return self.model
