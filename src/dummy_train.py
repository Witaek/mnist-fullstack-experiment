import torch
import mlflow
from dummy_model import DummyModel

model = DummyModel()
mlflow.set_experiment("dummy_experiment")
with mlflow.start_run():
    mlflow.log_param("model_type", "DummyModel")
    mlflow.log_metric("dummy_accuracy", 0.42)
print("Training complete.")
