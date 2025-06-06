import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalModule(nn.Module):
    """
    Convolutional module for extracting features from short windows (500 ms) of raw EEG.
    
    Args:
        n_time_steps: Number of time steps in the input
        n_in_channels: Number of input channels (default: 32 for EEG channels)
        n_hidden_channels: Number of hidden channels (default: 320)
        kernel_size: Kernel size for convolutions (default: 3)
        dilation: Dilation factor for convolutions (default: 3)
        out_dim: Output dimension (default: 2048)
        use_attention_pooling: Whether to use attention pooling (True) or simple pooling (False) (default: False)
        pooling_channels: Number of channels for pooling (used for both attention and simple pooling) (default: 64)
    """
    
    def __init__(
        self,
        n_time_steps: int,
        n_in_channels: int = 32,
        n_hidden_channels: int = 320,
        kernel_size: int = 3,
        dilation: int = 3,
        out_dim: int = 2048,
        use_attention_pooling: bool = False,
        pooling_channels: int = 64
    ):
        super(ConvolutionalModule, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.n_in_channels = n_in_channels
        self.n_hidden_channels = n_hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = kernel_size // 2 * dilation
        self.out_dim = out_dim
        self.use_attention_pooling = use_attention_pooling
        self.pooling_channels = pooling_channels
        
        # Initial Conv1d
        self.initial_conv = nn.Conv1d(
            in_channels=n_in_channels,
            out_channels=n_hidden_channels,
            kernel_size=1
        )
        
        # 5 Blocks
        self.blocks = nn.ModuleList([
            ConvBlock(
                n_hidden_channels=n_hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding
            ) for _ in range(5)
        ])
        
        # Pooling mechanism
        if use_attention_pooling:
            self.pooling = AttentionPooling(
                n_hidden_channels=n_hidden_channels,
                pooling_channels=pooling_channels,
                out_dim=out_dim
            )
        else:
            self.pooling = SimplePooling(
                n_hidden_channels=n_hidden_channels,
                n_time_steps=n_time_steps,
                pooling_channels=pooling_channels,
                out_dim=out_dim
            )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, n_in_channels, n_time_steps]
            
        Returns:
            Output tensor of shape [batch_size, out_dim]
        """
        # Initial convolution
        x = self.initial_conv(x)  # [B, n_hidden_channels, n_time_steps]
        
        # Pass through 5 blocks
        for block in self.blocks:
            x = block(x)  # [B, n_hidden_channels, n_time_steps]
            
        # Pooling (both mechanisms now output final dimension directly)
        x = self.pooling(x)  # [B, out_dim]
        
        return x


class ConvBlock(nn.Module):
    """
    A single convolutional block containing:
    - 2 Convolutions with BatchNorm, GELU, and skip connections
    - 1 Convolution with GLU activation
    """
    
    def __init__(self, n_hidden_channels, kernel_size, dilation, padding):
        super(ConvBlock, self).__init__()
        
        # First convolution with skip connection
        self.conv1 = nn.Conv1d(
            in_channels=n_hidden_channels,
            out_channels=n_hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden_channels)
        self.gelu1 = nn.GELU()
        
        # Second convolution with skip connection
        self.conv2 = nn.Conv1d(
            in_channels=n_hidden_channels,
            out_channels=n_hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden_channels)
        self.gelu2 = nn.GELU()
        
        # Third convolution with GLU
        self.conv3 = nn.Conv1d(
            in_channels=n_hidden_channels,
            out_channels=2 * n_hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.glu = nn.GLU(dim=1)  # GLU splits channels in half
        
    def forward(self, x):
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor of shape [batch_size, n_hidden_channels, n_time_steps]
            
        Returns:
            Output tensor of shape [batch_size, n_hidden_channels, n_time_steps]
        """
        # First convolution with skip connection
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)
        out = out + identity  # Skip connection
        
        # Second convolution with skip connection
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        out = out + identity  # Skip connection
        
        # Third convolution with GLU
        out = self.conv3(out)
        out = self.glu(out)  # GLU reduces channels back to n_hidden_channels
        
        return out


class SimplePooling(nn.Module):
    """
    Simple pooling mechanism that:
    1. Applies 1x1 convolution to reduce channels
    2. Flattens the result
    3. Applies linear transformation to output dimension
    """
    
    def __init__(self, n_hidden_channels, n_time_steps, pooling_channels, out_dim):
        super(SimplePooling, self).__init__()
        
        # 1x1 convolution to reduce channels
        self.conv = nn.Conv1d(
            in_channels=n_hidden_channels,
            out_channels=pooling_channels,
            kernel_size=1
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Linear layer
        self.linear = nn.Linear(n_time_steps * pooling_channels, out_dim)
        
    def forward(self, x):
        """
        Forward pass through simple pooling.
        
        Args:
            x: Input tensor of shape [batch_size, n_hidden_channels, n_time_steps]
            
        Returns:
            Output tensor of shape [batch_size, out_dim]
        """
        # Apply 1x1 convolution
        x = self.conv(x)  # [B, pooling_channels, n_time_steps]
        
        # Flatten
        x = self.flatten(x)  # [B, pooling_channels * n_time_steps]
        
        # Linear transformation
        x = self.linear(x)  # [B, out_dim]
        
        return x


class AttentionPooling(nn.Module):
    """
    Attention pooling mechanism that:
    1. Expands channels
    2. Computes attention scores
    3. Applies weighted pooling
    4. Applies final linear transformation to output dimension
    """
    
    def __init__(self, n_hidden_channels, pooling_channels, out_dim):
        super(AttentionPooling, self).__init__()
        
        # Expand channels
        self.x_exp = nn.Conv1d(
            in_channels=n_hidden_channels,
            out_channels=pooling_channels,
            kernel_size=1
        )
        
        # Attention score computation
        self.attn_conv = nn.Conv1d(
            in_channels=pooling_channels,
            out_channels=1,
            kernel_size=1
        )
        
        # Final linear layer
        self.linear = nn.Linear(pooling_channels, out_dim)
        
    def forward(self, x):
        """
        Forward pass through attention pooling.
        
        Args:
            x: Input tensor of shape [batch_size, n_hidden_channels, n_time_steps]
            
        Returns:
            Output tensor of shape [batch_size, out_dim]
        """
        # Expand channels
        x_exp = self.x_exp(x)  # [B, pooling_channels, n_time_steps]
        
        # Compute attention scores
        attn_scores = self.attn_conv(x_exp).squeeze(1)  # [B, n_time_steps]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_time_steps]
        
        # Apply attention weights
        x_weighted = x_exp * attn_weights.unsqueeze(1)  # [B, pooling_channels, n_time_steps]
        
        # Pool along time dimension
        pooled = x_weighted.sum(dim=-1)  # [B, pooling_channels]
        
        # Final linear transformation
        output = self.linear(pooled)  # [B, out_dim]
        
        return output
