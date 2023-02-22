import matplotlib.pyplot as plt
import numpy as np
import torch

from helpers.dataset_helpers import get_mnist_sets


def display_gallery(images, title, nb_columns=3, nb_rows=3):
    nb_images = len(images)
    nb_pages = nb_images // (nb_columns * nb_rows)
    if nb_images % (nb_columns * nb_rows) != 0:
        nb_pages += 1

    maximum = nb_pages * nb_columns * nb_rows
    if nb_images < maximum:
        rest = maximum - nb_images
        padding = [(np.zeros_like(images[0][0]), "None") for i in range(rest)]
        images.extend(padding)

    images = np.array(images)
    images = images.reshape(nb_pages, nb_rows, nb_columns, 2)

    for page in range(nb_pages):
        fig, axis = plt.subplots(nb_rows, nb_columns)
        fig.suptitle("page: {} - {}".format(page + 1, title))

        for i in range(nb_rows):
            for j in range(nb_columns):
                tmp_img = np.transpose(images[page][i][j][0], (1, 2, 0))
                axis[i, j].imshow(tmp_img.squeeze())
                axis[i, j].title.set_text(images[page][i][j][1])

    plt.show()


def get_miss_classified(model, iterator, miss_classified_count, true_index=1):
    model.eval()

    diff = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            y_true = batch[true_index]

            if len(y_true.shape) == 1:
                y_true = y_true.type('torch.LongTensor')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                src = src.cuda()
                y_true = y_true.cuda()

            y_pred = model(src)

            if type(y_pred) is tuple:
                y_pred, _ = y_pred

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

            bad_classification_indexes = np.flatnonzero(y_pred.argmax(axis=1) != y_true)
            bad_images = np.take(src, bad_classification_indexes, axis=0)
            bad_prediction = np.take(y_pred.argmax(axis=1), bad_classification_indexes, axis=0)
            bad_true_label = np.take(y_true, bad_classification_indexes, axis=0)

            new_pairs = [(i, "P:{} T:{}".format(j, k)) for i, j, k in zip(bad_images, bad_prediction, bad_true_label)]
            diff.extend(new_pairs)

    return diff


if __name__ == "__main__":
    print("Hello")
    train_set, test_set, _ = get_mnist_sets()

    arr = []
    for i in range(9):
        tmp = test_set[int(i)]
        tmp = (tmp[0].detach().cpu().numpy(), tmp[1])
        arr.append(tmp)

    display_gallery(arr, "bob", 3, 2)
    print("Done")
