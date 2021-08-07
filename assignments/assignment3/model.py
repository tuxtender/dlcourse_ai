import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # Create necessary layers
        image_width, image_height, n_channels = input_shape

        self.Conv1 = ConvolutionalLayer(n_channels, conv1_channels, 3, 1)
        self.Relu1 = ReLULayer()
        self.MaxPool1 = MaxPoolingLayer(4, 4)
        self.Conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.Relu2 = ReLULayer()
        self.MaxPool2 = MaxPoolingLayer(4, 4)
        self.Flatten = Flattener()
        self.FC = FullyConnectedLayer(2*2*conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        params = self.params()

        for param in params.values():
            param.grad *= 0.

        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        output = self.Conv1.forward(X)
        output = self.Relu1.forward(output)
        output = self.MaxPool1.forward(output)
        output = self.Conv2.forward(output)
        output = self.Relu2.forward(output)
        output = self.MaxPool2.forward(output)
        output = self.Flatten.forward(output)
        output = self.FC.forward(output)

        loss, d_pred = softmax_with_cross_entropy(output, y)

        d_out = self.FC.backward(d_pred)
        d_out = self.Flatten.backward(d_out)
        d_out = self.MaxPool2.backward(d_out)
        d_out = self.Relu2.backward(d_out)
        d_out = self.Conv2.backward(d_out)
        d_out = self.MaxPool1.backward(d_out)
        d_out = self.Relu1.backward(d_out)
        d_out = self.Conv1.backward(d_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        output = self.Conv1.forward(X)
        output = self.Relu1.forward(output)
        output = self.MaxPool1.forward(output)
        output = self.Conv2.forward(output)
        output = self.Relu2.forward(output)
        output = self.MaxPool2.forward(output)
        output = self.Flatten.forward(output)
        output = self.FC.forward(output)

        pred = np.argmax(output, axis=1)

        return pred

    def params(self):
        result = {}

        # Aggregate all the params from all the layers
        # which have parameters
        result["Conv1 W"] = self.Conv1.params()["W"]
        result["Conv1 B"] = self.Conv1.params()["B"]
        result["Conv2 W"] = self.Conv2.params()["W"]
        result["Conv2 B"] = self.Conv2.params()["B"]
        result["FC W"] = self.FC.params()["W"]
        result["FC B"] = self.FC.params()["B"]

        return result
