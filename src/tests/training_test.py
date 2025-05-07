import torch
from src.training.trainer import train_one_epoch
from src.tests.utils import get_dummy_dataloader, DummyModel, dummy_cfg
import pytest 
from unittest.mock import patch
import mlflow

mlflow.set_experiment("test_experiment")

def test_training_runs_without_crashing():
    dataloader = get_dummy_dataloader()
    model = DummyModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = []
    cfg = dummy_cfg()

    train_one_epoch(dataloader, model, loss_fn, metrics, optimizer, "cpu", epoch=0, cfg=cfg)

def test_model_weights_change_after_training():
    dataloader = get_dummy_dataloader()
    model = DummyModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = []
    cfg = dummy_cfg()

    before = {k: v.clone() for k, v in model.state_dict().items()}
    train_one_epoch(dataloader, model, loss_fn, metrics, optimizer, "cpu", epoch=0, cfg=cfg)
    after = model.state_dict()

    assert any(not torch.equal(before[k], after[k]) for k in before), "Model weights did not change"

def test_empty_dataloader_does_not_crash():
    dataloader = get_dummy_dataloader(num_samples=0)
    model = DummyModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = dummy_cfg()

    try:
        train_one_epoch(dataloader, model, loss_fn, [], optimizer, "cpu", epoch=0, cfg=cfg)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError on empty dataloader")



@patch("mlflow.log_metric")
def test_mlflow_logging_called(mock_log_metric):
    dataloader = get_dummy_dataloader()
    model = DummyModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = dummy_cfg()

    train_one_epoch(dataloader, model, loss_fn, [], optimizer, "cpu", epoch=0, cfg=cfg)

    assert mock_log_metric.called, "mlflow.log_metric was not called"
