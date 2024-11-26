from torch import nn

class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, time, features)
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