from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment
from helpers.dataset_helpers import get_mnist_sets
from models.conv_embedding import ConvEmbedding
from models.vit import ViT


def main_vit():
    summary = SummaryWriter()
    patch_size = (4, 4)
    embedding_size = 64

    embedding_layer = ConvEmbedding(1, patch_size=patch_size, embedding_size=embedding_size)
    model = ViT(embedding_layer=embedding_layer,
                img_size=(1, 28, 28),
                nb_output=10,
                nb_encoder_blocks=6,
                nb_heads=4)

    nb_epochs = 30
    batch_size = 128
    learning_rate = 0.0005
    datasets = get_mnist_sets()
    run_specific_experiment(summary, model, datasets, nb_epochs, batch_size, learning_rate)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_vit()
    print("Done")
