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
    batch_size = batch_index.shape[0]
    loss = - np.sum(np.log(probs[batch_index, target_index]))/batch_size

    return loss


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
    # Copy from previous assignment
    grad = reg_strength*2*W
    loss = reg_strength*np.sum(W**2)

    return loss, grad


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
    # Copy from the previous assignment
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


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # Copy from the previous assignment
        result = np.maximum(X, 0)
        self.X = X

        return result

    def backward(self, d_out):
        # Copy from the previous assignment
        d_result = d_out
        d_result[self.X < 0] = 0

        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Copy from the previous assignment
        self.X = X
        result = X @ self.W.value + self.B.value

        return result

    def backward(self, d_out):
        # Copy from the previous assignment
        
        self.W.grad += self.X.T @ d_out
        self.B.grad += np.mean(d_out, axis=0) * d_out.shape[0]

        d_input = d_out @ self.W.value.T

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        filter_height = self.filter_size
        filter_width = self.filter_size
        out_height = height - filter_height - 2*self.padding + 1
        out_width = width - filter_width - 2*self.padding + 1
        result = np.empty((batch_size, out_height, out_width, self.out_channels))
        self.X = X

        # Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # Implement forward pass for specific location
                I = X[:, y:y+filter_height, x:x+filter_width].reshape(batch_size, -1)
                W = self.W.value.reshape(-1, self.out_channels)
                result[:, y, x] = I @ W + self.B.value

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_input =  np.zeros((batch_size, height, width, channels))
        d_out_column = d_out.reshape(batch_size, -1)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                I = self.X[:, y:y+self.filter_size, x:x+self.filter_size]
                I_column = I.reshape(batch_size, -1)

                d_W = I_column.T @ d_out_column
                
                self.W.grad += d_W.reshape(self.out_channels, self.filter_size,
                                           self.filter_size, self.in_channels)
                self.B.grad += np.sum(d_out, axis=0).reshape(-1)

                W = self.W.value.T.reshape(out_channels, -1)

                d_field_column = d_out_column @ W
               
                d_field_inverse = d_field_column.reshape(batch_size, channels,
                                                         self.filter_size, self.filter_size)
                d_field = np.moveaxis(d_field_inverse.T, 3, 0)

                d_input[:, y:y+self.filter_size, x:x+self.filter_size, :] += d_field

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
