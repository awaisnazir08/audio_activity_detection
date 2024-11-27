from torch import nn
import torch

class LSTMSoundClassifier(nn.Module):
    """
    LSTM-based sound classifier model for sound event detection (e.g., cheering detection).

    Args:
        input_size (int): The number of input features for each time step (e.g., number of Mel-frequency bins).
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of LSTM layers.
        output_size (int): The number of output classes (e.g., 1 for binary classification).

    Attributes:
        lstm (nn.LSTM): The LSTM layer for sequence processing.
        fc (nn.Linear): The fully connected layer that maps the LSTM output to the final classification score.
        sigmoid (nn.Sigmoid): The sigmoid activation function for binary classification.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initializes the LSTM-based sound classifier with the given parameters.

        Args:
            input_size (int): Number of input features (e.g., Mel-frequency bins).
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output units (e.g., 1 for binary classification).
        """
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, input_size).

        Returns:
            torch.Tensor: Output tensor after passing through the LSTM, fully connected layer, and sigmoid activation.
                        The output will have shape (batch_size, time_steps, output_size).
        """
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)

        return self.sigmoid(out)



class CNNTransformerSoundClassifier(nn.Module):
    def __init__(self, input_size, num_heads, transformer_dim, cnn_filters=16, num_transformer_layers=2, output_size=1):
        super(CNNTransformerSoundClassifier, self).__init__()
        
        # 1D CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
            # nn.MaxPool1d(kernel_size=2),
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_filters,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Fully Connected Layer for classification
        self.fc = nn.Linear(cnn_filters, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, input_size, sequence_length)
        x = self.cnn(x)  # Output shape: (batch_size, cnn_filters, reduced_sequence_length)
        x = x.permute(0, 2, 1)  # Shape for Transformer: (batch_size, reduced_sequence_length, cnn_filters)
        x = self.transformer(x)  # Shape: (batch_size, reduced_sequence_length, cnn_filters)
        x = self.fc(x)  # Shape: (batch_size, reduced_sequence_length, output_size)
        return self.sigmoid(x)  # Shape: (batch_size, reduced_sequence_length, output_size)