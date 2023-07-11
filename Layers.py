import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError



class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    """
    np.dot means dot product.
    The dot product of two column matrices is multiplying the rows together
    [a] dot [c] is equal to [a*c]
    [b]     [d]             [b*d]
    """

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.

    def backward_propagation(self, output_error, learning_rate):
        # deel E / deel X
        input_error = np.dot(output_error, self.weights.T)

        # deel E / deel W
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error

        return input_error


class ActivationLayer(Layer):
    # Pass in the activation methods as Python functions
    # Derivative means partial derivative in this context
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    # returns the activated input
    def forward_propagation(self, input_data):
        # Previous neuron outputs
        self.input = input_data

        # Put input neuron(Previous neuron outputs) into the activation function
        self.output = self.activation(self.input)
        return self.output

    # Derivative means partial derivative in this context
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error