# -*- encoding: utf-8 -*-

import numpy as np

# if you implement ConvLayer by convolve function, you will use the following code.
from scipy.signal import fftconvolve as convolve

class ConvLayer():
	"""
	2D convolutional layer.
	This layer creates a convolution kernel that is convolved with the layer
	input to produce a tensor of outputs.
	Arguments:
		inputs: Integer, the channels number of input.
		filters: Integer, the number of filters in the convolution.
		kernel_size: Integer, specifying the height and width of the 2D convolution window (height==width in this case).
		pad: Integer, the size of padding area.
		trainable: Boolean, whether this layer is trainable.
	"""
	def __init__(self, inputs,
	             filters,
	             kernel_size,
	             pad,
	             trainable=True):
		self.inputs = inputs
		self.filters = filters
		self.kernel_size = kernel_size
		self.pad = pad
		assert pad < kernel_size, "pad should be less than kernel_size"
		self.trainable = trainable

		self.XavierInit()

		self.grad_W = np.zeros_like(self.W)
		self.grad_b = np.zeros_like(self.b)

	def XavierInit(self):
		raw_std = (2 / (self.inputs + self.filters))**0.5
		init_std = raw_std * (2**0.5)

		self.W = np.random.normal(0, init_std, (self.filters, self.inputs, self.kernel_size, self.kernel_size))
		self.b = np.random.normal(0, init_std, (self.filters,))

	def forward(self, Input, **kwargs):
		'''
		forward method: perform convolution operation on the input.
		Agrs:
			Input: A batch of images, shape-(batch_size, channels, height, width)
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
		backward method: perform back-propagation operation on weights and biases.
		Args:
			delta: Local sensitivity, shape-(batch_size, filters, output_height, output_width)
		Return:
			delta of previous layer
		'''
		############################################################################
	    # TODO: Put your code here
		# Calculate self.grad_W, self.grad_b, and return the new delta.


	    ############################################################################
