from datetime import datetime
import os
import socket
from shutil import copy

import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error

import numpy as np
from tensorboardX import SummaryWriter  # pylint: disable=import-error

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("using cpu")


print_every = 50


def create_log_dir(my_path, comment=""):

    """
   create folder for logs
   """

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        my_path, "runs", current_time + "_" + socket.gethostname() + comment
    )
    os.mkdir(log_dir)
    return log_dir


def accuracy(val_set, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (x, y) in val_set:

            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return {'acc':acc,'num_samples':num_samples}


def train(train_set, val_set, model, optimizer,log_dir, epochs=1,long_patience=True):
    model = model.to(device=device)
    log_dir = log_dir
    writer = SummaryWriter(log_dir=log_dir)
    if long_patience:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=40)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15)  
    for e in range(epochs):
        print("epoch: %d" % e)
        for i, (x, y) in enumerate(train_set):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = nn.functional.cross_entropy(scores, y)
            # loss = nn.functional.nll_loss(scores, y)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % print_every == 0:
                print("Iteration %d, loss = %.4f" % (i, loss.item()))
                acc_dict = accuracy(val_set, model)
                acc = acc_dict['acc']
                num_samples = acc_dict['num_samples']
                print(
                    "Got %d / %d correct (%.2f)" % (int(acc*num_samples), num_samples, 100 * acc)
                )
                lossi = np.float(loss.item())
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                writer.add_scalar("loss", lossi, i)
                writer.add_scalar("acc", acc, i)
                writer.add_scalar("lr", lr, i)
                print()
        scheduler.step(loss)
    
