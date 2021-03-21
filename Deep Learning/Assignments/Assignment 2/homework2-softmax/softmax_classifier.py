import numpy as np

def softmax(output):
  return np.transpose(np.exp(np.transpose(output)) / np.sum(np.exp(output), axis=1))

def ce_loss(output, label):
  return np.mean(-np.sum(label * np.log(output), axis=1))

def grad(input, output, label):
  return (-1/input.shape[0])*(np.transpose(input)@(label - output))

def softmax_classifier(w, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - w: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    # Output -> Softmax of W*X
    output = softmax(input @ w)
    # Loss -> Cross-entropy + Regularization
    loss = ce_loss(output, label) + (lamda/2)*np.linalg.norm(w)
    # Softmax gradient
    gradient = grad(input, output, label)
    # Predicted label
    prediction = np.argmax(output, axis=1)
    ############################################################################

    return loss, gradient, prediction
