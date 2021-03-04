import math
import argparse
import mnist_data_loader

import numpy as np

from math import log2
from tqdm import trange


class Args:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--max_epoch", type=int, default=100)
        self.parser = parser

    def args(self):
        return self.parser.parse_args()


class LogisticRegression:
    def __init__(self) -> None:
        # Weights -> shape = 785 (784 pixels + 1 bias)
        self.w = np.random.normal(size=785)
        self.lr = 0.01

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def loss(p, t):
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))

    @staticmethod
    def acc(p, t):
        p = [0 if px <= 0.5 else 1 for px in p]
        num_correct = 0
        for i in range(len(p)):
            if p[i] == t[i]:
                num_correct += 1

        return num_correct * 100 / len(p)

    def forward(self, input_id):
        return self.sigmoid(input_id @ self.w)

    def backward(self, i, p, t):
        return np.mean(i * (p - t)[:, None], axis=0)

    def update_weights(self, i, p, t):
        gradient = self.backward(i, p, t)
        self.w -= self.lr * gradient


def load_mnist():
    mnist = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)

    # Training dataset
    train_set = mnist.train
    # Binarize the train labels
    train_set.labels[train_set.labels == 3] = 0
    train_set.labels[train_set.labels == 6] = 1

    # Test dataset
    test_set = mnist.test
    # Binarize the test labels
    test_set.labels[test_set.labels == 3] = 0
    test_set.labels[test_set.labels == 6] = 1

    # append 1 to features to integrate bias term
    print(train_set.images)
    bias = np.ones(train_set.images.shape[0])
    train_set._images = np.c_[train_set.images, bias]
    
    return train_set, test_set

def train(model, train_set, args):
    print(">>> Training")
    for _ in trange(0, args.max_epoch):
        iter_per_batch = train_set.num_examples // args.batch_size
        num =0
        sum = 0
        for batch_id in trange(0, iter_per_batch):
            batch = train_set.next_batch(args.batch_size)
            input_ids, labels = batch
            # prediction
            output = model.forward(input_ids)
            # calculate the loss (and accuracy)
            loss = model.loss(output, labels)
            acc = model.acc(output, labels)
            sum += acc
            num +=1
            # update the weights
            model.update_weights(input_ids, output, labels)

            # print(f"loss: {loss}")
            # print(f"acc: {acc}")

        print(sum/num)


if __name__ == "__main__":
    train_set, test_set = load_mnist()
    args = Args().args()
    model = LogisticRegression()
    train(model, train_set, args)
