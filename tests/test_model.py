import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTNet

def test_model_parameters():
    model = MNISTNet()
    num_params = model.get_num_parameters()
    assert num_params < 20000, f"Model has {num_params} parameters, should be less than 20000"
    print(f"Parameter count test passed: {num_params} parameters")

def test_input_output_dimensions():
    model = MNISTNet()
    # Test input shape
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28)
    output = model(test_input)
    
    assert output.shape == (batch_size, 10), f"Output shape is {output.shape}, should be {(batch_size, 10)}"
    print("Input/Output dimensions test passed")

def test_model_architecture():
    model = MNISTNet()
    
    assert model.has_batch_norm(), "Model should use batch normalization"
    assert model.has_dropout(), "Model should use dropout"
    assert model.has_fc_layer(), "Model should have fully connected layers"
    print("Architecture components test passed")

def test_model_forward():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10)
        print("Forward pass test passed")
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {str(e)}")

if __name__ == "__main__":
    print("Running model tests...")
    test_model_parameters()
    test_input_output_dimensions()
    test_model_architecture()
    test_model_forward()
    print("All tests passed!") 