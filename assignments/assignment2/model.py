import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # Create necessary layers
        self.input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.reLU_layer = ReLULayer()
        self.hidden_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        params_dict = self.params()

        for param in params_dict.values():
            param.grad *= 0.
        
        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        input_layer_output = self.input_layer.forward(X)
        input_hidden_layer = self.reLU_layer.forward(input_layer_output)
        output_layer = self.hidden_layer.forward(input_hidden_layer)

        loss, d_prediction = softmax_with_cross_entropy(output_layer, y)

        d_hidden_layer = self.hidden_layer.backward(d_prediction)
        d_reLU_layer = self.reLU_layer.backward(d_hidden_layer)
        d_input_layer = self.input_layer.backward(d_reLU_layer)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for param in params_dict.values():
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            loss += loss_l2
            param.grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused

        input_layer_output = self.input_layer.forward(X)
        input_hidden_layer = self.reLU_layer.forward(input_layer_output)
        output_layer = self.hidden_layer.forward(input_hidden_layer)

        pred = np.argmax(output_layer, axis=1)

        return pred

    def params(self):
        result = {}

        # Implement aggregating all of the params
        result["Input layer W"] = self.input_layer.params()["W"]
        result["Input layer B"] = self.input_layer.params()["B"]
        result["Hidden layer W"] = self.hidden_layer.params()["W"]
        result["Hidden layer B"] = self.hidden_layer.params()["B"]

        return result
