""" ReLU Layer """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		Applies the rectified linear unit function element-wise: relu(x) = max(x, 0)
		"""
		self.trainable = False # no parameters

	def forward(self, Input):

		############################################################################
	    # TODO: Put your code here
		# Apply ReLU activation function to Input, and return results.


	    ############################################################################


	def backward(self, delta):

		############################################################################
	    # TODO: Put your code here
		# Calculate the gradient using the later layer's gradient: delta


	    ############################################################################
