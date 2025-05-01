def test_dummy_model_runs():
    from dummy_model import DummyModel
    model = DummyModel()
    out = model.forward(torch.randn(1, 1, 28, 28))
    assert out.shape == (1, 10)
