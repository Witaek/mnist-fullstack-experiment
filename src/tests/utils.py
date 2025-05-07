import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

def get_dummy_dataloader(batch_size=2, num_samples=10):
    x = torch.randn(num_samples, 1, 32,32)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(DummyModel, self).__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=2, stride = 4 )  # Output: 4×13×13
        self.fc = torch.nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))      # -> 4×13×13
        print(x.shape)
        x = x.view(x.size(0), -1)     # Flatten
        x = self.fc(x)                # -> logits for each class
        return x
    
def dummy_cfg():
    return SimpleNamespace(
        training=SimpleNamespace(
            logging_step=2,
            verbose_step=5,
            log=True,
            learning_rate=0.01,
            epochs=1,
        )
    )
