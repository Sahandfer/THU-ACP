""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit = logit
		self.gt = gt
		self.output = np.transpose(np.exp(np.transpose(logit)) / np.sum(np.exp(logit), axis=1))
		self.loss = np.mean(-np.sum(gt * np.log(self.output), axis=1))
		self.acc = np.sum(np.argmax(self.output, axis=1) == np.argmax(gt, axis=1)) / self.output.shape[0]
	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		batch_size = self.logit.shape[0]
		return (self.output - self.gt) * (1./batch_size)
	    ############################################################################
