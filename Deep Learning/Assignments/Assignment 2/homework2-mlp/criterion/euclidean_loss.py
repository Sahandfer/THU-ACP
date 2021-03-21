""" Euclidean Loss Layer """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.acc and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.logit = logit
		self.gt = gt
		self.loss = np.sum((gt-logit)**2, axis=1) / 2
		self.acc = np.mean(np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1))) / 100
		self.acc = np.sum(np.argmax(logit, axis=1) == np.argmax(gt, axis=1)) / logit.shape[0]
	    ############################################################################
		
		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		batch_size = self.logit.shape[0]
		return (self.logit - self.gt) * (1./batch_size)
	    ############################################################################
