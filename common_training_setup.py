import os
from os import path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from helpers.metrics_helpers import arg_max_accuracy
from train_eval.eval import evaluate
from train_eval.training import train


def run_specific_experiment(summary, model, datasets, nb_epochs, batch_size, learning_rate):
    train_set, test_set = datasets
    model_name = model.__class__.__name__

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

    train(model,
          train_set,
          optimizer,
          criterion,
          None,
          batch_size,
          nb_epochs,
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