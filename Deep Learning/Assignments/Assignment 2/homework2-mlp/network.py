""" Network Class """

class Network():
	def __init__(self):
		self.layerList = []
		self.numLayer = 0

	def add(self, layer):
		self.numLayer += 1
		self.layerList.append(layer)

	def forward(self, x):
		# forward layer by layer
		for i in range(self.numLayer):
			x = self.layerList[i].forward(x)
		return x

	def backward(self, delta):
		# backward layer by layer
		for i in reversed(range(self.numLayer)): # reversed
			delta = self.layerList[i].backward(delta)
