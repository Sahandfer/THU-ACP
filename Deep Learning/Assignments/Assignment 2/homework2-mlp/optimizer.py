""" Optimizer Class """

import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay

	# One backpropagation step, update weights layer by layer
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:

				############################################################################
			    # TODO: Put your code here
				# Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
				# Do not forget the weightDecay term.


			    ############################################################################

				# Weight update
				layer.W += layer.diff_W
				layer.b += layer.diff_b
