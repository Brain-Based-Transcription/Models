from convolutional_module import ConvolutionalModule


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_breakdown(model):
    """Print detailed parameter breakdown for the ConvolutionalModule."""
    print("Parameter Breakdown:")
    print("-" * 50)
    
    # Initial convolution
    initial_params = count_parameters(model.initial_conv)
    print(f"Initial Conv1d: {initial_params:,} parameters")
    
    # Blocks
    total_block_params = 0
    for i, block in enumerate(model.blocks):
        block_params = count_parameters(block)
        total_block_params += block_params
        print(f"Block {i+1}: {block_params:,} parameters")
        
        # Breakdown within each block
        conv1_params = count_parameters(block.conv1)
        bn1_params = count_parameters(block.bn1)
        conv2_params = count_parameters(block.conv2)
        bn2_params = count_parameters(block.bn2)
        conv3_params = count_parameters(block.conv3)
        
        print(f"  - Conv1 + BN1: {conv1_params + bn1_params:,} parameters")
        print(f"  - Conv2 + BN2: {conv2_params + bn2_params:,} parameters")
        print(f"  - Conv3 (GLU): {conv3_params:,} parameters")
    
    print(f"Total Blocks: {total_block_params:,} parameters")
    
    # Pooling mechanism
    pooling_params = count_parameters(model.pooling)
    
    if model.use_attention_pooling:
        print(f"Attention Pooling: {pooling_params:,} parameters")
        
        # Breakdown of attention pooling
        x_exp_params = count_parameters(model.pooling.x_exp)
        attn_conv_params = count_parameters(model.pooling.attn_conv)
        linear_params = count_parameters(model.pooling.linear)
        print(f"  - Channel expansion: {x_exp_params:,} parameters")
        print(f"  - Attention conv: {attn_conv_params:,} parameters")
        print(f"  - Final linear: {linear_params:,} parameters")
    else:
        print(f"Simple Pooling: {pooling_params:,} parameters")
        
        # Breakdown of simple pooling
        conv_params = count_parameters(model.pooling.conv)
        linear_params = count_parameters(model.pooling.linear)
        print(f"  - 1x1 Conv: {conv_params:,} parameters")
        print(f"  - Linear: {linear_params:,} parameters")
    
    # Total
    total_params = count_parameters(model)
    print("-" * 50)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    
    # Memory estimation (rough)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Estimated model size: {param_size_mb:.2f} MB")


# Test both pooling mechanisms
print("="*60)
print("SIMPLE POOLING (use_attention_pooling=False)")
print("="*60)

mod_simple = ConvolutionalModule(n_time_steps=32, use_attention_pooling=False)

print("Model Architecture:")
print(mod_simple)
print("\n")

print_parameter_breakdown(mod_simple)

print("\n\n")

print("="*60)
print("ATTENTION POOLING (use_attention_pooling=True)")
print("="*60)

mod_attention = ConvolutionalModule(n_time_steps=32, use_attention_pooling=True, pooling_channels=1024)

print("Model Architecture:")
print(mod_attention)
print("\n")

print_parameter_breakdown(mod_attention)