from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment
from helpers.dataset_helpers import get_mnist_sets
from models.vit import ViT


def main_vit():
    summary = SummaryWriter()
    model = ViT(img_size=(1, 28, 28),
                patch_size=(7,7),
                patch_hidden_size=32,
                nb_output=10,
                group_channel=False,
                nb_encoder_blocks=6,
                nb_heads=4)

    nb_epochs = 25
    batch_size = 256
    learning_rate = 0.0005
    datasets = get_mnist_sets()
    run_specific_experiment(summary, model, datasets, nb_epochs, batch_size, learning_rate)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_vit()
    print("Done")
