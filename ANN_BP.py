import pandas as pd
import random
import math
import numpy as np
import sys
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

# Function to initialize the neural network with random weights
def network_initialization(num_inputs, num_outputs, num_hidden_layers, neurons_per_layer):
    weight_list = []
    for idx in range(num_hidden_layers + 1):
        if idx == 0:
            hidden_layer = [[random.random() for _ in range(num_inputs + 1)] for _ in range(neurons_per_layer[idx])]
        elif idx == num_hidden_layers:
            hidden_layer = [[random.random() for _ in range(neurons_per_layer[idx - 1] + 1)] for _ in range(num_outputs)]
        else:
            hidden_layer = [[random.random() for _ in range(neurons_per_layer[idx - 1] + 1)] for _ in range(neurons_per_layer[idx])]
        weight_list.append(hidden_layer)
    return weight_list

# Activation function to calculate the output of a neuron
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return transfer(activation)

# Sigmoid function to squash the activation to a value between 0 and 1
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

# Forward propagation function to calculate the output for each neuron in the network
def forward_propagate(weight_list, data_row):
    inputs = data_row[:-1]
    neuron_outputs = []
    neuron_outputs.append(data_row[:-1])
    for layer_idx in range(len(weight_list)):
        new_inputs = []
        for neuron_idx in range(len(weight_list[layer_idx])):
            activation = activate(weight_list[layer_idx][neuron_idx], inputs)
            new_inputs.append(activation)
        inputs = new_inputs
        neuron_outputs.append(new_inputs)
    return neuron_outputs

# Backward propagation function to update the weights of the network based on the error
def backward_propagate(outputs, expected_output, weight_list, learning_rate):
    delta_list = []
    for output_layer_idx in reversed(range(len(outputs))):
        new_delta = []
        if output_layer_idx == len(outputs) - 1:
            for idx in range(len(outputs[output_layer_idx])):
                delta = outputs[output_layer_idx][idx] * (1 - outputs[output_layer_idx][idx]) * (
                    expected_output[idx] - outputs[output_layer_idx][idx])
                new_delta.append(delta)
            delta_list = new_delta
        elif output_layer_idx == 0:
            for idx in range(len(outputs[output_layer_idx])):
                for weight_idx in range(len(weight_list[output_layer_idx])):
                    weight_list[output_layer_idx][weight_idx][idx] += learning_rate * delta_list[weight_idx] * \
                                                                       outputs[output_layer_idx][idx]
            for weight_idx in range(len(weight_list[output_layer_idx])):
                weight_list[output_layer_idx][weight_idx][-1] += learning_rate * delta_list[weight_idx]
        else:
            for idx in range(len(outputs[output_layer_idx])):
                total_sum = 0
                for weight_idx in range(len(weight_list[output_layer_idx])):
                    total_sum += weight_list[output_layer_idx][weight_idx][idx] * delta_list[weight_idx]
                    weight_list[output_layer_idx][weight_idx][idx] += learning_rate * delta_list[weight_idx] * \
                                                                        outputs[output_layer_idx][                                                                       idx]
                delta = outputs[output_layer_idx][idx] * (1 - outputs[output_layer_idx][idx]) * total_sum
                new_delta.append(delta)
            for weight_idx in range(len(weight_list[output_layer_idx])):
                weight_list[output_layer_idx][weight_idx][-1] += learning_rate * delta_list[weight_idx]
            delta_list = new_delta

# Function to train the neural network
def train_network(weight_list, training_data_set, learning_rate, num_iterations, num_outputs):
    for iteration in range(num_iterations):
        sum_error = 0
        for row in training_data_set:
            outputs = forward_propagate(weight_list, row)
            expected = [0 for _ in range(num_outputs)]
            expected[int(row[-1]) - 1] = 1
            sum_error += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
            backward_propagate(outputs, expected, weight_list, learning_rate)
        sum_error /= len(training_data_set)
        print('Iteration=%d, Error=%.8f' % (iteration + 1, sum_error))
        precised_error = '%.8f' % sum_error
        if float(precised_error) == 0.0:
            break

# Function to print the weights of the network
def print_weights(weights):
    for layer in range(len(weights)):
        print("Layer " + str(layer) + ":")
        for col in range(len(weights[layer][0])):
            neuron_weights = []
            for row in range(len(weights[layer])):
                neuron_weights.append(weights[layer][row][col])
            if col == len(weights[layer][0]) - 1:
                print("\t Bias Term :" + str(neuron_weights))
            else:
                print("\t Neuron " + str(col + 1) + " : " + str(neuron_weights))

# Function to get the max output from the last layer
def max_output(last_outputs):
    actuals = [0 for _ in range(len(last_outputs))]
    index = last_outputs.index(max(last_outputs))
    actuals[index] = 1
    return actuals

# Function to test the model and return the error and accuracy
def test_the_model(data_set, weight_list, num_outputs):
    sum_error = 0
    count = 0
    for data in data_set:
        outputs = forward_propagate(weight_list, data)
        expected = [0 for _ in range(num_outputs)]
        expected[int(data[-1]) - 1] = 1
        sum_error += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
        actuals = max_output(outputs[len(outputs) - 1])
        if actuals == expected:
            count += 1
    sum_error /= len(data_set)
    accuracy = count / len(data_set)
    outs = []
    outs.append(sum_error)
    outs.append(accuracy)
    return outs

# Import required libraries
import numpy as np
from numpy import exp, array, random, dot

# Define the NeuralNetwork class
class NeuralNetwork():
    def __init__(self):
        random.seed(1) # Seed the random number generator
        self.synaptic_weights = 2 * random.random((3, 1)) - 1 # Initialize the synaptic weights

    # Define the sigmoid activation function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Define the derivative of the sigmoid function
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    # Make predictions using the trained neural network
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

# Main function to run the example neural network
if __name__ == "__main__":
    neural_network = NeuralNetwork() # Instantiate the neural network

    print("Forward Propagate: ")
    print(neural_network.synaptic_weights) # Print initial synaptic weights

    # Define the training data
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("Back Propagate Error: ")
    print(neural_network.synaptic_weights) # Print updated synaptic weights

    # Test the neural network with new data
    print("ANN Train[1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0]))) # Make a prediction

