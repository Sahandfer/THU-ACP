import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from tensorflow.examples.tutorials.mnist import input_data

def shuffle(x, l):
    randIndx = np.arange(len(x))
    np.random.shuffle(randIndx)
    x, l = x[randIndx], l[randIndx]
    
    return x, l

# Load the MNIST dataset in binary form (one-hot)
def load_dataset():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    x_train, l_train = mnist.train.images, mnist.train.labels

    return x_train, l_train


# Save a generated image (imported from zhusuan/example)
def save_image(x, filename, shape=(10, 10), scale_each=False, transpose=False):
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print("Shape too small to contain all images")
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype="uint8")
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h : (i + 1) * h, j * w : (j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)
