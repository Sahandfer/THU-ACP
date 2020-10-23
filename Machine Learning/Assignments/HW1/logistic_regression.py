import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, cross_validation
from sklearn.datasets import load_svmlight_file

# Global variables
acc_train = {}
acc_test = {}


def load_dataset():
    # Load the training and testing data from the given files.
    # 124 features = 123 given features from the dataset + 1 bias
    x_train, y_train = load_svmlight_file("Dataset/a9a", n_features=124)
    x_test, y_test = load_svmlight_file("Dataset/a9a.t", n_features=124)

    # Change the data type to np.ndarray and add the bias term as the 124th feature
    x_train, x_test = x_train.toarray(), x_test.toarray()
    x_train[:, 123], x_test[:, 123] = 1, 1

    # Make the output binary
    y_train = [0 if sample == -1 else 1 for sample in y_train]
    y_test = [0 if sample == -1 else 1 for sample in y_test]

    return x_train, y_train, x_test, y_test


def sigmoid(val):
    return 1 / (1+np.exp(-val))


def update_weights(x, w, y, _lambda=0):
    mu = sigmoid(x @ w)
    R = np.diag(np.multiply(mu, 1-mu))
    # Hessian matrix
    H = - (np.transpose(x)@R@x) - (_lambda*np.identity(124))
    # Gradient loss term
    G = - np.transpose(x)@(mu - y) - _lambda * w
    # Weight update
    w_new = w - np.linalg.pinv(H) @ G

    return w_new


def accuracy(x, w, y, epoch, mode):
    p = sigmoid(x@w)
    p = [0 if px < 0.5 else 1 for px in p]

    num_correct = 0
    for i in range(len(p)):
        if (p[i] == y[i]):
            num_correct += 1

    acc = num_correct * 100 / len(p)
    
    print("Accuracy:", acc)
    if (mode == "train"):
        acc_train[epoch] = acc
    else:
        acc_test[epoch] = acc


def precision():
    print("precise")


def train_model(x_train, w, y_train,_lambda, epoch=0):
    print("Training Epoch ==> %d" % epoch)

    if (epoch != 0):
        w = update_weights(x_train, w, y_train, _lambda)

    accuracy(x_train, w, y_train, epoch, "train")

    return w


def test_model(x_test, w, y_test, epoch=0):
    print("Testing Epoch ==> %d" % epoch)
    accuracy(x_test, w, y_test, epoch, "")


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()
    w = np.zeros((1, 124), dtype='float64')[0]
    
    epochs = 7
    _lambda = 20

    w = train_model(x_train, w, y_train, _lambda, 0)
    test_model(x_test, w, y_test, 0)
    
    for epoch in range(1, epochs+1):
        # Train the model based on the train data
        w = train_model(x_train, w, y_train, _lambda, epoch)
        # Test the model
        test_model(x_test, w, y_test, epoch)

    num_zeros = 0
    