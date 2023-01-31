from datetime import timedelta

import torch
import time

from helpers.metrics_helpers import arg_max_accuracy
from train_eval.eval import evaluate


def train(model, train_dataset, optimizer,
          criterion, scheduler, batch_size,
          n_epochs, shuffle, summary, save_file,
          early_stop=None, train_ratio=0.85, true_index=1):
    tc = int(len(train_dataset) * train_ratio)
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()
        x, y = torch.utils.data.random_split(train_dataset, [tc, len(train_dataset) - tc])
        train_iterator = torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=4, shuffle=shuffle)
        valid_iterator = torch.utils.data.DataLoader(y, batch_size=batch_size, num_workers=4, shuffle=shuffle)

        metrics = {"loss": criterion, "acc": arg_max_accuracy}
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, true_index=true_index)
        _, _, validation_metrics = evaluate(model, valid_iterator, metrics, true_index=true_index)
        if scheduler is not None:
            scheduler.step(validation_metrics["loss"])

        end_time = time.time()

        delta_time = timedelta(seconds=(end_time - start_time))

        if validation_metrics["loss"] < best_valid_loss:
            best_valid_loss = validation_metrics["loss"]
            if save_file is not None:
                torch.save(model, save_file)

        validation_str = metrics_to_string(validation_metrics, "val")
        print("Current Epoch: {} -> train_eval time: {}\n\tTrain Loss: {:.3f} - {}".format(epoch + 1, delta_time, train_loss, validation_str))
        summary.add_scalar("Loss/train", train_loss, epoch)
        log_metrics_in_tensorboard(summary, validation_metrics, epoch, "val")
        summary.flush()

        if early_stop is not None:
            if early_stop.should_stop(validation_metrics):
                break

    return best_valid_loss


def train_epoch(model, iterator, optimizer, criterion, true_index = 1):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch[0]
        y_true = batch[true_index]

        if len(y_true.shape) == 1:
            y_true = y_true.type('torch.LongTensor')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            src = src.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()

        y_pred = model(src)
        loss = criterion(y_pred, y_true)

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def log_metrics_in_tensorboard(summary, metrics, epoch, prefix):
    for k, val in metrics.items():
        summary.add_scalar("{}/{}".format(prefix, k), val, epoch + 1)


def metrics_to_string(metrics, prefix):
    res = []
    for k, val in metrics.items():
        res.append("{} {}:{:.3f}".format(prefix, k, val))

    return " ".join(res)
