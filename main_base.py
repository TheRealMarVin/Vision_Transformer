from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment
from helpers.dataset_helpers import get_mnist_sets
from models.mlp import MLP


def main_mlp():
    summary = SummaryWriter()
    model = MLP(input_dim=28*28, hidden_size=500, out_size=10)
    nb_epochs = 5
    batch_size = 128
    learning_rate = 0.00005
    datasets = get_mnist_sets()
    run_specific_experiment(summary, model, datasets, nb_epochs, batch_size, learning_rate)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_mlp()
    print("Done")
