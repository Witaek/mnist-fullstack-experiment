import torch
import pytest
from src.models import LeNet5
from src.tests.utils import DummyModel

model_classes = [DummyModel, LeNet5]

@pytest.mark.parametrize("ModelClass", model_classes)
def test_model_output_shape(ModelClass):
    model = ModelClass(num_classes = 10)
    dummy_input = torch.randn(8, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (8,10), f"{ModelClass.__name__} output shape mismatch !"

@pytest.mark.parametrize("ModelClass", model_classes)
def test_model_runs_forward(ModelClass):
    model = ModelClass(num_classes = 10)
    dummy_input = torch.randn(8, 1, 32, 32)

    try:
        _ = model(dummy_input)
    except Exception as e:
        pytest.fail(f"{ModelClass.__name__} failed forward pass: {e}")