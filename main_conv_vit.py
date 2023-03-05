from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment
from helpers.dataset_helpers import get_mnist_sets
from models.embeddings.conv_embedding import ConvEmbedding
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

    print(model)
    train_config_file = "config/training_params.ini"
    train_set, test_set, image_size = get_mnist_sets()
    run_specific_experiment(summary, model, (train_set, test_set), train_config_file)
    summary.close()


if __name__ == "__main__":
    print("Hello")
    main_vit()
    print("Done")
