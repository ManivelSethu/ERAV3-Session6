import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTNet

def test_model_parameters():
    print("\nTesting model parameter count...")
    model = MNISTNet()
    num_params = model.get_num_parameters()
    assert num_params < 20000, f"Model has {num_params} parameters, should be less than 20000"
    print(f"✓ Parameter count test passed: Model has {num_params:,} parameters (target: <20,000)")

def test_input_output_dimensions():
    print("\nTesting input/output dimensions...")
    model = MNISTNet()
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    
    assert output.shape == (batch_size, 10), f"Output shape is {output.shape}, should be {(batch_size, 10)}"
    print(f"✓ Input/Output dimensions test passed:")
    print(f"  - Input shape: {test_input.shape} (batch_size, channels, height, width)")
    print(f"  - Output shape: {output.shape} (batch_size, num_classes)")

def test_model_architecture():
    print("\nTesting model architecture components...")
    model = MNISTNet()
    
    assert model.has_batch_norm(), "Model should use batch normalization"
    assert model.has_dropout(), "Model should use dropout"
    assert model.has_fc_layer(), "Model should have fully connected layers"
    print("✓ Architecture components test passed:")
    print("  - Has Batch Normalization: Yes")
    print("  - Has Dropout layers: Yes")
    print("  - Has Fully Connected layers: Yes")

def test_model_forward():
    print("\nTesting model forward pass...")
    model = MNISTNet()
    model.eval()  # Set to evaluation mode
    batch_size = 4  # Use batch size > 1
    x = torch.randn(batch_size, 1, 28, 28)
    try:
        with torch.no_grad():  # Disable gradient computation
            output = model(x)
        assert output.shape == (batch_size, 10)
        print("✓ Forward pass test passed:")
        print("  - Model successfully processes input")
        print("  - Output shape is correct")
        print("  - No errors in forward propagation")
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Running MNIST Model Tests")
    print("="*50)
    test_model_parameters()
    test_input_output_dimensions()
    test_model_architecture()
    test_model_forward()
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50 + "\n") 