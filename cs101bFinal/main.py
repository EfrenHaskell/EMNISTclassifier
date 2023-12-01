import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from net_utils import *


def load_data() -> (DataLoader, DataLoader):
    # downloads EMNIST training and test set and produces two data loaders
    train_set = torchvision.datasets.EMNIST(root="data", split="balanced", train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.EMNIST(root="data", split="balanced", train=False, transform=transforms.ToTensor(), download=True)
    train_loader: DataLoader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=100, shuffle=True)
    return train_loader, test_loader


def train(train_loader: DataLoader):
    # loops over input and expected values sampled from train_loader
    for batch, data in enumerate(train_loader, 0):
        input_vals, expected = data
        pass


def test(test_loader: DataLoader):
    pass


if __name__ == '__main__':
    epochs: int = 1
    train_data, test_data = load_data()
    # outer loop for multiple training epochs
    for epoch in range(epochs):
        train(train_data)
        test(test_data)
