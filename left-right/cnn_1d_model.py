import torch
import torch.nn as nn
from torchinfo import summary


class CNN1D(nn.Module):
    def __init__(self, 
                 input_channels=19,
                 num_samples=600,
                 sample_freq=200,
                 channels=[95, 100, 100, 100, 100],
                 kernel_size=10,
                 dropout=0.25,
                 activation=nn.GELU,
                 pooling=nn.MaxPool1d,
                 pooling_kernel_size=2,
                 num_conv_blocks=3,
                ):
        super(CNN1D, self).__init__()
        
        # Store parameters
        self.input_channels = input_channels
        self.num_samples = num_samples
        self.sample_freq = sample_freq
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.pooling = pooling
        self.pooling_kernel_size = pooling_kernel_size
        self.num_conv_blocks = num_conv_blocks

        # Temporal Convolution Block
        self.temporal_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.channels[0],
                kernel_size=self.sample_freq // 2,
                groups=self.input_channels,
                padding=(self.sample_freq // 2 - 1) // 2
            ),
            nn.BatchNorm1d(self.channels[0]),
            self.activation(),
            nn.Dropout1d(p=self.dropout),
        )
        
        # Spatial Convolution Block
        self.spatial_block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                kernel_size=1
            ),
            nn.BatchNorm1d(self.channels[1]),
            self.activation(),
            nn.Dropout1d(p=self.dropout),
        )
        
        # Conv Blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(self.num_conv_blocks):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.channels[i+1],
                    out_channels=self.channels[i+2],
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(self.channels[i+2]),
                self.activation(),
                nn.Dropout1d(p=self.dropout),
                self.pooling(kernel_size=self.pooling_kernel_size)
            )
            self.conv_blocks.append(block)
        
        # Final Linear Layer - LazyLinear automatically infers input size
        self.linear = nn.LazyLinear(2)
    
    def forward(self, x):
        # x shape: (batch_size, input_channels, num_samples)
        
        # Temporal Convolution
        x = self.temporal_block(x)
        
        # Spatial Convolution
        x = self.spatial_block(x)
        
        # 3 Blocks of Conv1d
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten
        x = x.flatten(start_dim=1)
        
        # Final Linear Layer
        x = self.linear(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Basic test
    model = CNN1D()
    x = torch.randn(2, 19, 600)
    y = model(x)  # Initialize LazyLinear
    
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print("Model works correctly!\n")
    
    # Print detailed model summary
    summary(model, input_size=x.shape)
