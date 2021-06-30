import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # Implement softmax
    # Your final implementation shouldn't have any loops
   
    if predictions.ndim == 1:
      predictions -= np.max(predictions)
      predictions_exp = np.exp(predictions)
      probs = predictions_exp/np.sum(predictions_exp)
      return probs

    batch_max = np.max(predictions, axis=1)[:, np.newaxis]
    predictions -= batch_max
    predictions_exp = np.exp(predictions)
    batch_sum = np.sum(predictions_exp, axis=1)
    probs = predictions_exp/batch_sum[:, np.newaxis]

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # Implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
      loss = - np.log(probs[target_index])
      return loss

    batch_index = np.arange(probs.shape[0])
    loss = - np.sum(np.log(probs[batch_index, target_index]))/batch_index.shape[0]

    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # Implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions.copy())
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()

    if predictions.ndim == 1:
      dprediction[target_index] -= 1
      return loss, dprediction

    batch_size = target_index.shape[0]
    batch_index = np.arange(batch_size)
    dprediction[batch_index, target_index] -= 1
    dprediction /= batch_size

    return loss, dprediction

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # Implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    grad = reg_strength*2*W
    loss = reg_strength*np.sum(W**2)

    return loss, grad

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = np.maximum(X, 0)
        self.X = X

        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out
        d_result[self.X < 0] = 0

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result = X @ self.W.value + self.B.value

        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += self.X.T @ d_out

        # TODO: Figure out what's it meaning. Guess equation by gradient_check output result.
        self.B.grad += np.mean(d_out, axis=0) * d_out.shape[0]

        d_input = d_out @ self.W.value.T

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B,}
