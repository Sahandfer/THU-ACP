# -*- encoding: utf-8 -*-

import numpy as np

class MaxPoolingLayer():
	def __init__(self, kernel_size, pad):
		'''
		This class performs max pooling operation on the input.
		Args:
			kernel_size: The height/width of the pooling kernel.
			pad: The width of the pad zone.
		'''

		self.kernel_size = kernel_size
		self.pad = pad
		self.trainable = False

	def forward(self, Input, **kwargs):
		'''
		This method performs max pooling operation on the input.
		Args:
			Input: The input need to be pooled.
		Return:
			The tensor after being pooled.
		'''
		############################################################################
	    # TODO: Put your code here
		# Apply convolution operation to Input, and return results.
		# Tips: you can use np.pad() to deal with padding.
		self.Input = Input
		input_after_pad = np.pad(Input, ((0,), (0,), (self.pad,), (self.pad,)), mode='constant', constant_values=0)

	    ############################################################################

	def backward(self, delta):
		'''
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate and return the new delta.


	    ############################################################################
