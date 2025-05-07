import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def create_mnist_dataloaders(data_path, train_val_split, batch_size, seed):

    if os.path.exists(f"{data_path}/MNIST"):
        print("MNIST Dataset already exists ! Loading...")
    else:
        print(f"No existing MNIST dataset found. Downloading to {data_path}...")

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Training set
    train_dataset = datasets.MNIST(
        root=data_path,         # Folder to store/download the dataset
        train=True,            # Load training set
        transform=transform,   # Apply the transform
        download=True          # Download if not already downloaded
    )

    # Test set
    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        transform=transform,
        download=True
    )

    total_train_size = len(train_dataset)
    train_size = int(train_val_split * total_train_size)
    eval_size = total_train_size - train_size

    train_subset, eval_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed)
    )


    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader   