

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    # Add a new layer to the network
    def add(self, layer):
        self.layers.append(layer)

    # Set the loss to use
    def use(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    # Predict output using the network with custom input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # Run network on all pieces
        for i in range(samples):
            # Forward Propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # Train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # Training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward prop
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Find loss for display only
                err += self.loss(y_train[j], output)

                # Back prop
                error = self.loss_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
