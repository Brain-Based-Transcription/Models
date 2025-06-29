import torch
import torch.nn as nn

class ConvLUNModel(nn.Module):
    def __init__(
        self,
        num_channels: int = 19,
        num_time_steps: int = 600,
        channels: list[int] = [25, 25, 50, 100, 100],
        temporal_kernel_1: int = 100,
        temporal_kernel_2: int = 10,
        dropout_rate: float = 0.25,
        activation: nn.Module = nn.GELU,
        pooling: nn.Module = nn.MaxPool2d,
    ):
        super(ConvLUNModel, self).__init__()

        # Variables
        self.temporal_kernel_1 = temporal_kernel_1
        self.temporal_kernel_2 = temporal_kernel_2
        self.num_channels = num_channels
        self.num_time_steps = num_time_steps
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.pooling = pooling
        self.channels = channels

        # Temporal Block
        self.temporal_block = nn.Sequential(
            nn.Conv2d(1, self.channels[0], (1, self.temporal_kernel_1), padding=(0, self.temporal_kernel_1 // 2)),
            nn.BatchNorm2d(self.channels[0]),
            self.activation(),
            nn.Dropout2d(p=self.dropout_rate)
        )

        # Spatial Block
        self.spatial_block = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], (self.num_channels, 1)),
            nn.BatchNorm2d(self.channels[1]),
            self.activation(),
            nn.Dropout2d(p=self.dropout_rate),
            self.pooling((1, 3))
        )

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], (1, self.temporal_kernel_2), padding=(0, self.temporal_kernel_2 // 2)),
            nn.BatchNorm2d(self.channels[2]),
            self.activation(),
            nn.Dropout2d(p=self.dropout_rate),
            self.pooling((1, 3))
        )

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(self.channels[2], self.channels[3], (1, self.temporal_kernel_2), padding=(0, self.temporal_kernel_2 // 2)),
            nn.BatchNorm2d(self.channels[3]),
            self.activation(),
            nn.Dropout2d(p=self.dropout_rate),
            self.pooling((1, 3))
        )

        # Conv Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(self.channels[3], self.channels[4], (1, self.temporal_kernel_2), padding=(0, self.temporal_kernel_2 // 2)),
            nn.BatchNorm2d(self.channels[4]),
            self.activation(),
            nn.Dropout2d(p=self.dropout_rate),
            self.pooling((1, 3))
        )

        self.flatten = nn.Flatten()
        
        # Calculate the size of the input to the linear layer dynamically
        final_size = self.num_time_steps // (3 ** 4)
        self.fc = nn.Linear(self.channels[-1] * final_size, 2)


    def forward(self, x):
        # Input shape: (batch_size, 1, num_channels, num_time_steps)
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Instantiate the model
    model = ConvLUNModel()
    print(model)

    # Create a dummy input tensor to test the model
    # batch_size=1, channels=1, num_channels=19, num_time_steps=600
    dummy_input = torch.randn(1, 1, 19, 600)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Expected output shape: (1, 2)
    assert output.shape == (1, 2)
    print("Model test passed!")
