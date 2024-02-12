import torch.nn as nn


class ImageRegressionCNN(nn.Module):
    """
    A convolutional neural network for image regression tasks.

    This network is designed to take in a single-channel image
    and output a single continuous value, making it suitable for
    regression problems involving images, such as estimating a
    numeric value from a visual input.

    The architecture includes several convolutional layers,
    LeakyReLU activations, and max pooling layers for feature
    extraction, followed by fully connected layers for regression.
    The network uses LeakyReLU activations for intermediate layers
    to introduce non-linearity and a ReLU activation for the penultimate layer.

    """
    def __init__(self):
        super(ImageRegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the correct input size
        self.fc1_input_size = 64 * 64 * 64
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.act2(self.conv2_5(x))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act2(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(self.act3(x))
        return x
