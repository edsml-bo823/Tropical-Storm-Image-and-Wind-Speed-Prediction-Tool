import torch
import numpy as np
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value
    and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


class ConvLSTMCell(nn.Module):
    """
    Implements a Convolutional LSTM cell.

    Combines LSTM with convolutional layers for
    spatiotemporal data processing.
    Useful in tasks like video frame prediction and
    weather forecasting.

    Attributes:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden state
        channels.
        kernel_size (int): Convolutional kernel size.
        padding (int): Padding for convolution, based
        on kernel size.
        conv (nn.Conv2d): Convolutional layer for input
        and hidden state.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden state channels.
        kernel_size (int): Convolutional kernel size.

    Methods:
        forward(input_tensor, cur_state):
            Advances the cell's state.
            Args:
                input_tensor (Tensor): Input tensor (batch,
                channels, height, width).
                cur_state (tuple): Current hidden and cell
                states (h_cur, c_cur).
            Returns:
                tuple: Next hidden and cell states (h_next,
                c_next).

        init_hidden(batch_size, image_size):
            Initializes hidden and cell states to zeros.
            Args:
                batch_size (int), image_size (tuple): Batch
                size and image dimensions.
            Returns:
                tuple: Initialized hidden and cell states.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=self.input_channels +
                              self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        combined_conv = F.elu(combined_conv)  # Applying ELU activation

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_channels,
                            height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels,
                            height, width, device=device))


class ConvEncoder(nn.Module):
    """
    A convolutional encoder module with a ConvLSTM cell.

    This encoder is designed for processing sequential
    spatial data. It applies a ConvLSTM cell
    followed by batch normalization and dropout for
    regularization.

    Attributes:
        conv_lstm1 (ConvLSTMCell): A ConvLSTM cell
        for sequential processing.
        batch_norm (nn.BatchNorm2d): Batch normalization layer.
        dropout (nn.Dropout2d): Dropout layer for regularization.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels
        for ConvLSTM.
        kernel_size (int): Kernel size for ConvLSTM.
        dropout_prob (float, optional): Dropout probability.
        Default is 0.7.

    Methods:
        forward(input_tensor):
            Processes the input through the ConvLSTM cell
            and applies batch normalization and dropout.
            Args:
                input_tensor (Tensor): Input tensor with
                shape (batch_size, seq_len, channels, height, width).
            Returns:
                tuple: Last state of the sequence and
                the final hidden and cell states.
    """
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, dropout_prob=0.7):
        super(ConvEncoder, self).__init__()
        self.conv_lstm1 = ConvLSTMCell(input_channels,
                                       hidden_channels // 2, kernel_size)

        self.batch_norm = nn.BatchNorm2d(hidden_channels // 2)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()
        h, c = self.conv_lstm1.init_hidden(batch_size, (height, width))
        last_state = None

        for t in range(seq_len):
            h, c = self.conv_lstm1(input_tensor[:, t, :, :, :], (h, c))

            h = self.batch_norm(h)
            h = self.dropout(h)
            if t == seq_len - 1:
                last_state = h

        return last_state, (h, c)


class ConvDecoder(nn.Module):
    """
    A convolutional decoder module with a ConvLSTM cell.

    This decoder is intended for generating spatially
    coherent sequences from encoded representations.
    It uses a ConvLSTM cell and applies batch
    normalization, dropout, and a final convolution layer.

    Attributes:
        conv_lstm1 (ConvLSTMCell): ConvLSTM
        cell for sequential data generation.
        batch_norm (nn.BatchNorm2d):
        Batch normalization layer.
        conv (nn.Conv2d): Convolutional layer to generate output.
        dropout (nn.Dropout2d): Dropout layer for regularization.

    Args:
        hidden_channels (int): Number of hidden channels for ConvLSTM.
        output_channels (int): Number of output channels
        for the final convolution.
        kernel_size (int): Kernel size for ConvLSTM and final convolution.
        dropout_prob (float, optional): Dropout probability. Default is 0.7.

    Methods:
        forward(encoder_last_state, h, c, seq_len):
            Processes the input through the ConvLSTM cell and
            generates the output sequence.
            Args:
                encoder_last_state (Tensor): Last state of the encoder.
                h, c (Tensor): Initial hidden and cell states for ConvLSTM.
                seq_len (int): Length of the sequence to generate.
            Returns:
                Tensor: Generated output tensor.
    """
    def __init__(self, hidden_channels, output_channels,
                 kernel_size, dropout_prob=0.7):
        super(ConvDecoder, self).__init__()
        self.conv_lstm1 = ConvLSTMCell(hidden_channels // 2,
                                       hidden_channels // 2,
                                       kernel_size)
        self.batch_norm = nn.BatchNorm2d(hidden_channels // 2)
        self.conv = nn.Conv2d(hidden_channels // 2,
                              output_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, encoder_last_state, h, c, seq_len):
        outputs = []

        for t in range(seq_len):
            h, c = self.conv_lstm1(h, (h, c))
            h = self.batch_norm(h)
            if t == 0:
                h = h + encoder_last_state
            h = self.dropout(h)
            outputs.append(h)

        output = self.conv(outputs[-1])
        output = output.unsqueeze(1)

        return output


class Seq2SeqAutoencoder(nn.Module):
    """
    Implements a sequence-to-sequence autoencoder using
    convolutional LSTM cells.

    This autoencoder is designed for tasks that involve sequential
    and spatial data, such as video processing.
    It comprises a convolutional encoder and a convolutional decoder
    for encoding and decoding the input data.

    Attributes:
        encoder (ConvEncoder): The convolutional encoder module.
        decoder (ConvDecoder): The convolutional decoder module.

    Args:
        input_channels (int): Number of channels in the input data.
        hidden_channels (int): Number of hidden channels in the ConvLSTM cells.
        output_channels (int): Number of channels in the output data.
        kernel_size (int): Size of the kernel in the ConvLSTM cells.

    Methods:
        forward(input_tensor):
            Processes the input tensor through the encoder and decoder.
            Args:
                input_tensor (Tensor): The input tensor with shape
                (batch_size, seq_len, channels, height, width).
            Returns:
                Tensor: The decoded output tensor.
    """
    def __init__(self, input_channels, hidden_channels,
                 output_channels, kernel_size):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = ConvEncoder(input_channels,
                                   hidden_channels, kernel_size)
        self.decoder = ConvDecoder(hidden_channels,
                                   output_channels, kernel_size)

    def forward(self, input_tensor):
        encoder_last_state, (h, c) = self.encoder(input_tensor)
        output = self.decoder(encoder_last_state, h, c, seq_len=1)
        return output


class AutoencoderTrainer():
    """
    A class for training a sequence-to-sequence autoencoder model.

    This class handles the training and evaluation of a
    Seq2SeqAutoencoder model using given data loaders,
    and tracks the performance using the Structural
    Similarity Index (SSIM) as the loss function.

    Attributes:
        device (torch.device): The device (CPU or CUDA) for
        training and evaluation.
        model (Seq2SeqAutoencoder): The Seq2SeqAutoencoder model.
        train_loader (DataLoader): DataLoader for the
        training dataset.
        test_loader (DataLoader): DataLoader for the
        testing/validation dataset.
        optimizer (optim.Optimizer): Optimizer for model training.
        num_epochs (int): Number of training epochs.
        input_channels, hidden_channels, output_channels,
        kernel_size: Model parameters.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the
        testing/validation dataset.
        input_channels (int): Number of channels in the input data.
        hidden_channels (int): Number of hidden channels in the ConvLSTM cells.
        output_channels (int): Number of channels in the output data.
        kernel_size (int): Size of the kernel in the ConvLSTM cells.
        num_epochs (int): Number of training epochs.

    Methods:
        train():
            Trains the model for one epoch and returns
            the average loss.
            Returns:
                float: Average training loss for the epoch.

        evaluate():
            Evaluates the model on the test dataset and
            returns the average loss.
            Returns:
                float: Average validation loss.

        train_model():
            Trains and evaluates the model for the specified number of epochs.
            Returns:
                Seq2SeqAutoencoder: The trained model.
    """
    def __init__(self, model, train_loader, test_loader, input_channels,
                 hidden_channels, output_channels, kernel_size, num_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.num_epochs = num_epochs
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def train(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = outputs.squeeze(2)

            # Calculate SSIM loss
            loss = 1 - ssim(outputs, targets, data_range=255,
                            size_average=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), \
                                  targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze(2)

                # Calculate SSIM score
                ssim_score = ssim(outputs, targets,
                                  data_range=255, size_average=True)
                loss = 1 - ssim_score

                total_loss += loss.item()

        average_loss = total_loss / len(self.test_loader)
        return average_loss

    def train_model(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train()
            val_loss = self.evaluate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss},\
                    Validation loss: {val_loss}")
        return self.model
