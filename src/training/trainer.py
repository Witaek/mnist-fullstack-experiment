import torch
from src.training.training_utils import MetricTracker
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate


def train_one_epoch(dataloader, model, loss_function, metrics_functions, optimizer, device, epoch, cfg):
    model.train()

    # Handle empty dataloader
    if len(dataloader) == 0:
        print("Warning: Dataloader is empty. Skipping training epoch.")
        return


    logging_step = cfg.training.logging_step
    verbose_step = cfg.training.verbose_step
    log = cfg.training.log

    tracker = MetricTracker(metrics_functions, logging_step)


    

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        prediction = model(data)
        loss = loss_function(prediction, target)
        metrics = tracker.update(prediction, target, loss)

        loss.backward()
        optimizer.step() 

        if log:
            step = epoch * len(dataloader) + batch_idx
            tracker.log_batch(batch_idx, loss, metrics, step)
        
        if batch_idx % verbose_step == 0:
            metric_strs = [f"{metric.__name__}: {metrics[i]:.4f}" for i, metric in enumerate(metrics_functions)]
            print(f"Loss: {loss.item():.4f} | " + " | ".join(metric_strs))

    if log:
        tracker.log_epoch(epoch)

    print(f"Average Train loss: {tracker.total_loss / len(dataloader):.4f}")


@hydra.main(version_base='1.3.2', config_path=None, config_name="config")
def train(cfg: DictConfig):
    
    print(f"Instantiating model : {cfg.model._target_}")
    model = instantiate(cfg.model)
    print(f"OK\n")

    print(f"Instantiating loss function : {cfg.training.loss._target_}")
    loss_function = instantiate(cfg.training.loss)
    print("OK\n")

    print(f"Instantiating optimizer : {cfg.training.optimizer._target_}")
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    print("OK\n")

    print(f"Instantiating dataloaders...")
    train_dataloader, eval_dataloader, test_dataloader = instantiate(cfg.data.dataset)
    print("OK\n")

    device = cfg.training.device
    print(f"Using {device} for training !")
    model.to(device)

    for epoch in tqdm(range(cfg.training.num_epochs)):
        train_one_epoch()

if __name__ == "__main__":
    train()
    
