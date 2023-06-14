""" Dropout Layer """

import numpy as np


class DropoutLayer:
    def __init__(self, p):
        self.trainable = False
        self.p = p

    def forward(self, Input, is_training=True):

        ############################################################################
        # TODO: Put your code here
        if is_training:
            probs = np.random.rand(*Input.shape)
            q = 1 - self.p
            self.mask = probs < q
            return self.mask * Input / q
        else:
            return Input
            ############################################################################

    def backward(self, delta):

        ############################################################################
        # TODO: Put your code here
        q = 1 - self.p
        return self.mask * delta / q
        ############################################################################
