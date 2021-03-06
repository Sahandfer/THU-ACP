import argparse
import numpy as np
import mnist_data_loader
from tqdm import trange
import matplotlib.pyplot as plt


class Args:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--max_epoch", type=int, default=12)
        parser.add_argument("--print_stats", default=False, action="store_true")
        self.parser = parser

    def args(self):
        return self.parser.parse_args()


class LogisticRegression:
    def __init__(self, input_shape) -> None:
        # Weights -> shape = 785 (784 pixels + 1 bias)
        self.w = np.random.normal(size=input_shape)
        self.lr = 0.1

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def loss(p, t):
        p[p==0] = 1e-10
        return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
            

    @staticmethod
    def acc(p, t):
        p = [0 if px <= 0.5 else 1 for px in p]
        num_correct = 0
        for i in range(len(p)):
            if p[i] == t[i]:
                num_correct += 1

        return num_correct * 100 / len(p)

    @staticmethod
    def grad(i, p, t):
        return np.mean(i * (p - t)[:, None], axis=0)

    def forward(self, input_id):
        return self.sigmoid(input_id @ self.w)

    def update_weights(self, i, p, t):
        gradient = self.grad(i, p, t)
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
    bias = np.ones((train_set.images.shape[0], 1))
    train_set._images = np.hstack((train_set.images, bias))
    bias = np.ones((test_set.images.shape[0], 1))
    test_set._images = np.hstack((test_set.images, bias))

    return train_set, test_set


def train(model, train_set, args):
    print(">>> Training")
    train_loss = []
    train_acc = []
    for _ in trange(0, args.max_epoch):
        iter_num = train_set.num_examples // args.batch_size
        for _ in trange(0, iter_num):
            batch = train_set.next_batch(args.batch_size)
            input_ids, labels = batch
            # forward (prediction)
            output = model.forward(input_ids)
            # calculate the loss (cross-entropy loss)
            loss = model.loss(output, labels)
            # calculate the accuracy
            acc = model.acc(output, labels)
            # update the weights
            model.update_weights(input_ids, output, labels)
            # record the stats
            train_loss.append(loss)
            train_acc.append(acc)

            if (args.print_stats):
                print(f"loss: {loss}")
                print(f"acc: {acc}%")

    print(f"Training accuracy: {sum(train_acc)/(args.max_epoch*iter_num)}%")
    print(">>> End of Training")

    return train_loss, train_acc


def test(model, test_set, args):
    print(">>> Testing")
    iter_num = test_set.num_examples // args.batch_size
    test_acc = 0 
    for _ in trange(0, iter_num):
        batch = train_set.next_batch(args.batch_size)
        input_ids, labels = batch
        # prediction
        output = model.forward(input_ids)
        # calculate the accuracy
        acc = model.acc(output, labels)
        test_acc += acc

        if (args.print_stats):
            print(f"acc: {acc}%")

    test_acc /= iter_num
    print(f"Testing accuracy: {test_acc}%")
    print(">>> End of Testing")
    return test_acc

def plot_curves(train_loss, train_acc):
    fig, _ = plt.subplots()
    plt.plot(train_loss)
    plt.title("Training Loss Curve")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    # plt.show()
    fig.savefig("figures/loss.png", dpi=300)

    fig, _ = plt.subplots()
    plt.plot(train_acc)
    plt.title("Training Accuracy Curve")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Step")
    # plt.show()
    fig.savefig("figures/acc.png", dpi=300)
    


if __name__ == "__main__":
    train_set, test_set = load_mnist()
    args = Args().args()
    model = LogisticRegression(train_set.images.shape[1])
    train_loss, train_acc = train(model, train_set, args)
    test(model, test_set, args)
    plot_curves(train_loss, train_acc)
