import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim

import os
from os import path

import numpy as np

from torchvision import transforms

from helpers.metrics_helpers import arg_max_accuracy
from models.mlp import MLP
from models.vit import ViT
from train_eval.eval import evaluate
from train_eval.training import train


def get_mnist_sets(train_transform, test_transform):
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    return train_set, test_set

def get_cifar10_sets(train_transform, test_transform):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set


def run_specific_experiment(summary, model):
    n_epochs = 5
    batch_size = 128
    learning_rate = 0.00005

    model_name = "ViT"

    summary.add_hparams({"learning rate": learning_rate,
                         "batch size": batch_size,
                         "max epochs": n_epochs}, {})

    out_folder = "saved_models/{}".format(model_name)
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    save_file = "{}/best.model".format(out_folder)

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_set, test_set = get_mnist_sets(train_transform, test_transform)
    # train_set, test_set = get_cifar10_sets(train_transform, test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model,
          train_set,
          optimizer,
          criterion,
          None,
          batch_size,
          n_epochs,
          True,
          summary,
          save_file,
          early_stop=None,
          true_index=1)
    print('Finished Training')

    metrics = {"loss": criterion, "acc": arg_max_accuracy}
    y_pred, y_true, valid_loss = evaluate(model, test_loader, metrics)
    y_pred = np.array(y_pred).argmax(1)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def main_mlp():
    summary = SummaryWriter()
    model = MLP(input_dim=28*28, hidden_size=500, out_size=10)
    run_specific_experiment(summary, model)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_mlp()
    print("Done")
