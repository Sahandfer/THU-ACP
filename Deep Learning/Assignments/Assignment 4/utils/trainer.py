import torch
from tqdm import trange


def train(model, train_loader, optimizer, criterion, args):
    loss_and_acc_dict = {}
    for e in trange(args["max_epoch"]):
        train_loss = 0.0
        num_correct, num_total = 0, 0

        model.train()
        for x, y in train_loader:

            output = model(x)

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            pred = torch.max(output.data, 1)[1]
            num_correct += (pred == y).sum()
            num_total += args["batch_size"]

            train_loss += loss

        train_loss = train_loss / len(train_loader.dataset)
        acc = num_correct * 100 / num_total

        print(f"\n Epoch {e+1} --> Training Loss: {train_loss} | Accuracy: {acc}%\n")

        loss_and_acc_dict[e] = [train_loss, acc]

    return loss_and_acc_dict
