import os

import numpy as np
import zhusuan as zs
import tensorflow as tf
import tensorflow_datasets as datasets


def load_dataset():
    """ "
    Function to load the whole MNIST dataset as training data
    """
    image, label = datasets.as_numpy(
        datasets.load(
            "mnist",
            split="train",
            batch_size=-1,
            as_supervised=True,
        )
    )
    print("loaded dataset")

    return image, label


if __name__ == "__main__":
    print("Zhusuan")
    load_dataset()