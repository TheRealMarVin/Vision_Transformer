import os
from configparser import ConfigParser
from os import path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from helpers.metrics_helpers import arg_max_accuracy
from helpers.result_helpers import get_miss_classified, display_gallery
from train_eval.eval import evaluate
from train_eval.training import train, metrics_to_string


def run_specific_experiment(summary, model, datasets, train_config_file):
    train_set, test_set = datasets
    model_name = model.__class__.__name__

    train_config = ConfigParser()
    train_config.read(train_config_file)

    nb_epochs = train_config.getint("default", "nb_epochs")
    batch_size = train_config.getint("default", "batch_size")
    learning_rate = train_config.getfloat("default", "learning_rate")

    summary.add_hparams({"model_name": model_name,
                         "learning rate": learning_rate,
                         "batch size": batch_size,
                         "max epochs": nb_epochs}, {})

    out_folder = "saved_models/{}".format(model_name)
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    save_file = "{}/best.model".format(out_folder)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)

    train(model,
          train_set,
          optimizer,
          criterion,
          scheduler,
          batch_size,
          nb_epochs,
          True,
          summary,
          save_file,
          early_stop=None,
          true_index=1)
    print("Finished Training")

    metrics = {"loss": criterion, "acc": arg_max_accuracy}
    y_pred, y_true, valid_loss = evaluate(model, test_loader, metrics)
    y_pred = np.array(y_pred).argmax(1)

    bad_prediction_pairs = get_miss_classified(model, test_loader, 50)
    display_gallery(bad_prediction_pairs, "Bad prediction", nb_columns=3, nb_rows=3)
    print(metrics_to_string(valid_loss, "test"))
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))
