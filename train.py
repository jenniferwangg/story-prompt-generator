import warnings

import torch
from torch.utils.data import DataLoader

from callbacks.PBar import PBar
from helpers.train_utils import train_epoch, validate

warnings.simplefilter('ignore')


def set_train(model, train_dataset, val_dataset, loss_fn, optimizer, epochs, batch_size, callbacks, device):
    # Creating data loader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(
        '\nSize of training data: {} \nSize of validation data: {} \n'.format(len(train_dataset), len(val_dataset)))

    # Initiating Callbacks
    history, csv_logger, model_checkpoint, early_stopping = callbacks

    #
    total_training_batches = len(train_dataset)
    total_validation__batches = len(val_dataset)

    # Training starts
    for epoch in range(epochs):
        history.reset_batch_logs()

        # Initializing progress bar for training
        pbar_train = PBar(total=total_training_batches)
        pbar_train.write('\nEpoch {}/{}'.format(epoch + 1, epochs))
        pbar_train.set_desc('Train')

        # Training an epoch
        train_epoch(model, loss_fn, optimizer, train_loader, pbar_train, history, device)

        # logging train epoch values
        history.update_epoch(mode='training')
        pbar_train.set_epoch_postfix(history.train_epoch_logs, mode='training')
        pbar_train.close()

        # Setting validation mode
        model.eval()
        torch.set_grad_enabled(False)

        # Initializing progress bar for training
        pbar_val = PBar(total=total_validation__batches)
        pbar_val.set_desc('Val  ')

        # Performing validation
        validate(model, val_loader, loss_fn, pbar_val, history, device)

        # Logging val epoch values
        history.update_epoch(mode='validation')
        pbar_val.set_epoch_postfix(history.val_epoch_logs, mode='validation')
        pbar_val.close()

        final_logs = history.train_epoch_logs.copy()
        final_logs.update(history.val_epoch_logs)

        # Executing remaining callbacks
        # model_checkpoint.set_model(model)
        # model_checkpoint.make_checkpoint(epoch, final_logs)
        csv_logger.update(epoch, final_logs)
        early_stopping.check(epoch, final_logs)

    return 0