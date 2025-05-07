import numpy as np
import mlflow

class MetricTracker:
    def __init__(self, metrics_functions, log_every_batch=20, log_every_epoch=True):
        self.metrics_functions = metrics_functions
        self.log_every_batch = log_every_batch
        self.log_every_epoch = log_every_epoch
        self.metrics_values = np.zeros(len(metrics_functions))  # For storing cumulative metric values
        self.batch_count = 0
        self.total_loss = 0

    def update(self, predictions, targets, loss):
        """
        Update the metrics after every batch.
        """
        # Compute metrics for the current batch and accumulate them
        batch_metrics = np.array([metric(predictions, targets) for metric in self.metrics_functions])
        self.metrics_values += batch_metrics
        self.batch_count += 1
        self.total_loss += loss.item()
        return batch_metrics

    def log_batch(self, batch_idx, loss, metrics, step):
        """
        Log metrics for every batch (if needed).
        """
        if batch_idx % self.log_every_batch == 0:
            # Log loss to MLflow
            mlflow.log_metric("loss", loss.item(), step=step)
            
            # Log other metrics
            for idx, metric in enumerate(self.metrics_functions):
                mlflow.log_metric(f"{metric.__name__}_batch", metrics[idx], step=step)

    def log_epoch(self, epoch):
        """
        Log averaged metrics for the epoch (if needed).
        """
        if self.log_every_epoch:
            # Average metrics over the entire epoch
            average_metrics = self.metrics_values / self.batch_count
            mlflow.log_metric("avg_loss", self.total_loss / self.batch_count, step=epoch)
            
            for idx, metric in enumerate(self.metrics_functions):
                mlflow.log_metric(f"{metric.__name__}_epoch", average_metrics[idx], step=epoch)