import numpy as np
from numpy import array

# Initialize weights and biases
weights = []
biases = []

# Set initial parameters
input_dim = 8
output_dim = 4
learning_rate = 0.00002
regularization = 0.3
hidden_dim = 14
hidden_layers = 2
num_iterations = 5000

# Helper function to evaluate the total loss on the dataset
def calculate_total_error(X, y):
    # Forward propagation
    num_layers = len(weights) - 1
    num_samples = len(X)

    activations_input = {}
    activations_value = {}
    activations_input[0] = X.dot(weights[0]) + biases[0]
    activations_value[0] = np.tanh(activations_input[0])

    for i in range(num_layers - 1):
        activations_input[i + 1] = activations_value[i].dot(weights[i + 1]) + biases[i + 1]
        activations_value[i + 1] = np.tanh(activations_input[i + 1])

    activations_input[num_layers] = activations_value[num_layers - 1].dot(weights[num_layers]) + biases[num_layers]

    # Calculating the probability using softmax function
    softmax_output = np.exp(activations_input[num_layers])
    probs = softmax_output / np.sum(softmax_output, axis=1, keepdims=True)

    # Calculating the loss using cross entropy loss function
    correct_logprobs = -np.log(probs[range(num_samples), y])
    data_loss = np.sum(correct_logprobs)

    # Adding regularization to loss using L2-norm loss function
    reg_sum = np.sum(np.square(weights[0]))
    for i in range(num_layers):
        reg_sum += np.sum(np.square(weights[i + 1]))
    data_loss += regularization / 2 * (reg_sum)

    return data_loss / float(num_samples)

# X - features and Y - labels, and the number of passes.
def train_network(X, y, num_passes=20000):
    # Initialize the parameters to random values. We need to learn these.
    num_samples = len(X)
    # Set the bias to zero and the weights to random.
    np.random.seed(0)

    # Initializing weights and biases
    weights.append(np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
    biases.append(np.zeros((1, hidden_dim)))

    for i in range(hidden_layers - 1):
        weights.append(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        biases.append(np.zeros((1, hidden_dim)))

    weights.append(np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim))
    biases.append(np.zeros((1, output_dim)))

    # Update the network num_pass times
    for j in range(num_passes):
        activations_input = {}
        activations_value = {}

        # Forward propagation
        activations_input[0] = X.dot(weights[0]) + biases[0]
        activations_value[0] = np.tanh(activations_input[0])

        # Hidden activations
        for i in range(hidden_layers - 1):
            activations_input[i + 1] = activations_value[i].dot(weights[i + 1]) + biases[i + 1]
            activations_value[i + 1] = np.tanh(activations_input[i + 1])

        # Output nodes
        activations_input[hidden_layers] = activations_value[hidden_layers - 1].dot(weights[hidden_layers]) + biases[hidden_layers]

       

       

            # Calculating the probability of each class using the softmax function
        softmax_output = np.exp(activations_input[hidden_layers])
        probs = softmax_output / np.sum(softmax_output, axis=1, keepdims=True)

        # Backpropagation
        deltas = {}
        delta_weights = {}
        delta_biases = {}
        deltas[hidden_layers] = probs

        # Calculate the error (delta) i.e., the (probability of predicted class - 1)
        deltas[hidden_layers][range(num_samples), y] -= 1

        delta_weights[hidden_layers] = (activations_value[hidden_layers - 1].T).dot(deltas[hidden_layers])
        delta_biases[hidden_layers] = np.sum(deltas[hidden_layers], axis=0, keepdims=True)

        for i in reversed(range(hidden_layers - 1)):
            deltas[i + 1] = deltas[i + 2].dot(weights[i + 2].T) * (1 - np.power(activations_value[i + 1], 2))
            delta_weights[i + 1] = np.dot(activations_value[i].T, deltas[i + 1])
            delta_biases[i + 1] = np.sum(deltas[i + 1], axis=0)

        deltas[0] = deltas[1].dot(weights[1].T) * (1 - np.power(activations_value[0], 2))
        delta_weights[0] = np.dot(X.T, deltas[0])
        delta_biases[0] = np.sum(deltas[0], axis=0)

        # Add regularization terms
        for i in range(hidden_layers + 1):
            delta_weights[i] += regularization * weights[i]
            weights[i] += -learning_rate * delta_weights[i]
            biases[i] += -learning_rate * delta_biases[i]

        # Print the loss
        if j % 1000 == 0:
            total_error = calculate_total_error(X, y)
            print("Iteration %i: Error: %f" % (j, total_error))
            if total_error < 0.1:
                print("Error less than minimum, exiting...")
                break

    return


def test_network(validation_data, validation_label):
    valid_count = 0
    num_layers = len(W) - 1
    act_input = {}
    act_val = {}
    act_input[0] = validation_data.dot(W[0]) + b[0]
    act_val[0] = np.tanh(act_input[0])

    for i in range(num_layers - 1):
        act_input[i + 1] = act_val[i].dot(W[i + 1]) + b[i + 1]
        act_val[i + 1] = np.tanh(act_input[i + 1])

    act_input[num_layers] = act_val[num_layers - 1].dot(W[num_layers]) + b[num_layers]

    softmax_val = np.exp(act_input[num_layers])
    probs = softmax_val / np.sum(softmax_val, axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)

    for p in range(len(predictions)):
        if predictions[p] == validation_label[p]:
            valid_count += 1

    print("correctly predicted:", str(valid_count), " --> ", str(valid_count * 100.0 / len(validation_data)) + "%")

    return valid_count / len(validation_data)

def get_training_data():
    with open("train_data.txt") as data_file:
        features = []
        labels = []
        for line in data_file:
            line_data = [float(d) for d in line.split()]
            features.append(line_data[1:9])
            labels.append(int(line_data[9]) - 1)
    return np.array(features), np.array(labels)

def get_test_data():
    with open("test_data.txt") as data_file:
        features = []
        labels = []
        for line in data_file:
            line_data = [float(d) for d in line.split()]
            features.append(line_data[1:9])
            labels.append(int(line_data[9]) - 1)
    return np.array(features), np.array(labels)

def set_params(input_nodes=8, output_nodes=4, hidden_nodes=12, hidden_layers=2, alpha=0.00001, reg=0.1, passes=25000):
    global input_dim, output_dim, hidden_dim, hidden_layers, epsilon, reg_lambda, num_passes
    input_dim = input_nodes
    output_dim = output_nodes
    hidden_dim = hidden_nodes
    hidden_layers = hidden_layers
    epsilon = alpha
    reg_lambda = reg
    num_passes = passes


def main():
    # Get the data from file
    train_data, training_label = get_training_data()
    test_data, test_label = get_test_data()

    # Build the model using the entire train dataset
    print("Training the network...")
    train_network(train_data[:6000], training_label[:6000], num_passes)
    # Test the constructed network with test_data
    print("Training completed.")
    print("Testing the network...")
    test_network(train_data[:6000], training_label[:6000])
    test_network(test_data[6000:], test_label[6000:])

if __name__ == "__main__":
    main()