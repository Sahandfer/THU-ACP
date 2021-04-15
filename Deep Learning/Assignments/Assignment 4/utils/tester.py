import torch
from tqdm import trange


def test(model, test_loader, batch_size):
    num_correct, num_total = 0, 0

    model.eval()
    for x, y in test_loader:

        output = model(x)
        pred = torch.max(output.data, 1)[1]
        num_correct += (pred == y).sum()
        num_total += batch_size

    acc = num_correct * 100 / num_total

    print(f"\n Testing Accuracy: {acc}%\n")
