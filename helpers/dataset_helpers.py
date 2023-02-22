import torchvision
from torchvision import transforms


def get_mnist_sets():
    image_resolution = (1, 28, 28)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    return train_set, test_set, image_resolution


def get_cifar10_sets():
    image_resolution = (1, 32, 32)

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set, image_resolution
