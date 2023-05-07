import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in input, hidden, and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # Define the activation function (sigmoid)
        self.activation_function = lambda x : 1.0 / (1.0 + np.exp(-x))

    def train(self, features, targets):
        '''
        Train the network on a batch of features and targets.

        Arguments
        ---------
        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        '''
        Implement forward pass here

        Arguments
        ---------
        X: features batch
        '''
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden) # signals into the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from the hidden layer

        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into the final output layer
        final_outputs = final_inputs.copy() # signals from the final output layer

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
        Implement backpropagation

        Arguments
        ---------
        final_outputs: output from forward pass
        y: target (i.e. label) batch
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        '''
        error = y - final_outputs

        hidden_error = error * self.weights_hidden_to_output
        hidden_error = hidden_error.reshape(hidden_error.shape[0])

        output_error_term = error.copy()

        hidden_error_term = (hidden_outputs * (1 - hidden_outputs))
        hidden_error_term = hidden_error_term * hidden_error

        delta_weights_i_h += self.lr * hidden_error_term * X[:, None]
        delta_weights_h_o += self.lr * output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += delta_weights_h_o / float(n_records) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h / float(n_records

    def run(self, features):
    '''
    Run a forward pass through the network with input features
    
    Arguments
    ---------
    features: 1D array of feature values
    '''

    hidden_inputs = np.matmul(features, self.weights_input_to_hidden) # signals into hidden layer
    hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

    final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
    final_outputs = final_inputs.copy() # signals from final output layer

    return final_outputs




iterations = 8000
learning_rate = 0.25
hidden_nodes = 6
output_nodes = 1