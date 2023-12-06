import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from net_utils import *
import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# for converting label index to label char value
CONVERSIONS: list[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                          'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b',
                          'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
# RUNWITHLOG defines whether to print loss and accuracy log
RUNWITHLOG: bool = False


def load_data() -> tuple[DataLoader, DataLoader]:
    # downloads EMNIST training and test set and produces two data loaders
    # EMNIST includes 47 classes, there should be 131,600 characters in total
    # Dimensions should be 28 x 28
    # tensor dimensions: [100, 1, 28, 28]
    train_set = torchvision.datasets.EMNIST(root="data", split="balanced", train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.EMNIST(root="data", split="balanced", train=False, transform=transforms.ToTensor(), download=True)
    train_loader: DataLoader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=1)
    test_loader: DataLoader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=1)
    return train_loader, test_loader


def train(train_loader: DataLoader, net, loss_function, optimizer, epoch_num):
    # loops over input and expected values sampled from train_loader
    running_loss: float = 0.0
    net.train()
    for batch, data in enumerate(train_loader, 0):
        input_vals, expected = data
        optimizer.zero_grad()
        outputs = net(input_vals)
        loss = loss_function(outputs, expected)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0 and batch != 0 and RUNWITHLOG:
            print(f'[{epoch_num + 1}, {batch + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    print('Finished Training')


def test(test_loader: DataLoader, net, epoch_num: int):
    net.eval()
    accuracy = 0

    with torch.no_grad():
        for features, true_labels in test_loader:
            predictions = net(features)
            predictions = torch.max(predictions, 1)[1].data.squeeze()
            accuracy = (predictions == true_labels).sum().item() / float(true_labels.size(0))
    if RUNWITHLOG:
        print(f"Accuracy at Epoch {epoch_num + 1}: {accuracy*100}%")


def example_UI(test_loader: DataLoader, net):
    samples: int = 5
    with torch.no_grad():
        for features, true_labels in test_loader:
            if samples == 0:
                break
            predictions = net(features)
            predictions = torch.max(predictions, 1)[1].data.squeeze()
            plt.imshow(features[0].data[0], cmap='gray')
            label_num = true_labels[0]
            plt.title(f"Sample {6 - samples}")
            value: int = predictions[0].item()
            txt = f"Expected: {CONVERSIONS[label_num]} | Got: {CONVERSIONS[value]}"
            plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
            plt.show()
            samples -= 1
            print(f"labels: {label_num}, {value}")

if __name__ == '__main__':
    epochs: int = 1
    train_data, test_data = load_data()
    net = Net()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    # outer loop for multiple training epochs
    for epoch in range(epochs):
        train(train_data, net, loss_function, optimizer, epoch)
        test(test_data, net, epoch)
    example_UI(test_data, net)
