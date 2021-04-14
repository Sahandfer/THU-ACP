import torch
import torch.nn as nn
from tqdm import trange
import torch.optim as optim


def train(model, train_loader, optimizer, criterion, args):
    for e in trange(args["max_epoch"]):
        train_loss = 0.0

        model.train()
        for x, y in train_loader:

            output = model(x)

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss

        print(f">>> Training Loss at epoch {e+1} ---> {train_loss}")
