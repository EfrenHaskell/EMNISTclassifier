import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
# from net_utils import *

def load_data() -> (DataLoader, DataLoader):
    train_set = torchvision.datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.MNIST(root="data", train=False, transform=transforms.ToTensor(), download=True)
    train_loader: DataLoader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=100, shuffle=True)
    return train_loader, test_loader


def train(train_loader: DataLoader):
    for batch, data in enumerate(train_loader, 0):
        input_vals, expected = data
        pass


def test(test_loader: DataLoader):
    pass


if __name__ == '__main__':
    epochs: int = 1
    train_data, test_data = load_data()
    for epoch in range(epochs):
        train(train_data)
        test(test_data)
