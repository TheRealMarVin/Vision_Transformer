from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment
from helpers.dataset_helpers import get_mnist_sets
from models.dummy_models.mlp import MLP


def main_mlp():
    summary = SummaryWriter()
    model = MLP(input_dim=28*28, hidden_size=500, out_size=10)
    train_config_file = "config/training_params.ini"
    train_set, test_set, image_size = get_mnist_sets()
    run_specific_experiment(summary, model, (train_set, test_set), train_config_file)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_mlp()
    print("Done")
