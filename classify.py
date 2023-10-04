import torch
from torch import optim
from torchsummary import summary

import torch.nn as nn
from callbacks.csvlogger import CSVLogger
from callbacks.early_stopping import EarlyStopping
from callbacks.history import History
from callbacks.model_checkpoint import ModelCheckpoint
from config import *
from helpers.my_dataset import MyDataset
from helpers.net import Net
# from helpers.myresnet18 import MyResnet18
from helpers.utils import get_grads_status, make_trainable_false
from losses import LossFn
from train import train


def create_cb():
    callbacks = [History(), CSVLogger('./log/train.log'),
                 ModelCheckpoint('./saves/train_ep_{epoch:d}-loss_{loss:.3f}-val_loss_{val_loss:.3f}.pth',
                                 monitor='val_loss',
                                 save_best_only=False, save_state_dict_only=True), EarlyStopping(patience=5)]

    return callbacks


def main(dev):
    # torch.backends.cudnn.enabled = False

    train_dataset = MyDataset(train_data_path)
    val_dataset = MyDataset(val_data_path)

    # Callbacks initialization
    callbacks = create_callbacks()

    # Model initialization
    model = Net()

    model = model if device == torch.device('cpu') else model.cuda(device)
    summary(model, (3, 224, 224))

    # make trainable true or false
    # make_trainable_false(model, -2)

    # Get trainable status for all the layers
    get_grads_status(model)

    # Loss function
    loss_fn = LossFn()

    # Optimizer
    optimizer = optim.Adamax(model.parameters(), weight_decay=0.0001)
    # optimizer = optim.SGD(params=model.parameters(),lr=0.001,weight_decay=0.0001,momentum=0.9,nesterov=True)

    train(model, train_dataset, val_dataset, loss_fn, optimizer, epochs, batch_size, callbacks, device)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('\t\tTraining on GPU device:0')
    else:
        device = torch.device('cpu')
        print('\t\tTraining on CPU')

    main(device)