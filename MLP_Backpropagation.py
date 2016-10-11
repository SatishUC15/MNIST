from __future__ import division
import numpy as np
import random as rand
import math

activation_threshold = 0.90


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Sigmoid activation function
def sigmoid_activation(x):
    result = (1 / (1 + math.exp(-x)))
    # return result
    if result > activation_threshold:
        return 1
    else:
        return 0


# Function that returns the gradient i.e. derivative of sigmoid(x)
def gradient_sigmoid(x):
    result = sigmoid(x) / ((1 + sigmoid(x)) ** 2)
    return result
    # if result > activation_threshold:
    #     return 1
    # else:
    #     return 0


# Function to train a multi-layer perceptron and validate the model using cross validation
#       2D array  data                 data set with attributes and class labels
#       Integer   num_hidden_layers    number of hidden layers
#       1D array  num_hidden_layers    1D array containing the number of neurons in each hidden layer
#       Integer   num_output_neurons   number of output neurons
#       Float     learning_rate
#       Float     momentum
def train_and_classify(data, num_hidden_layers, num_hidden_neurons, num_output_neurons,
                       learning_rate, momentum):

    num_epochs = 80
    training_error = []

    # Partitioning training and test data
    num_records = np.size(data, 0)
    training_size = int(np.floor(0.8 * num_records))
    batch_size = int(np.floor(0.4 * training_size))

    training_data = data[0:training_size, :]
    test_data = data[training_size:num_records, :]

    # initialize weights - random values
    num_neurons = []
    num_neurons[:] = num_hidden_neurons[:]
    num_neurons.append(num_output_neurons)
    weights = {}
    changes = {}

    print "Initializing weights..."
    for layer in range(num_hidden_layers+1):
        weight_matrix = []
        changes_matrix = []
        for neuron_idx in range(num_neurons[layer]):
            row = []
            change = []
            if layer == 0:  # Initial layer will receive direct input signals
                for idx in range(len(data[0])):
                    row.append(rand.random()/1000)
                    change.append(0)

            else:  # The other layers will receive signals from the previous layer
                for idx in range(num_neurons[layer-1]+1):
                    row.append(rand.random()/1000)
                    change.append(0)
            weight_matrix.append(row)
            changes_matrix.append(change)
        weights[layer] = weight_matrix
        changes[layer] = changes_matrix

    delta = {}
    activation = {}
    summation = {}

    print "Training the multi-layer perceptron... "
    for epoch in range(num_epochs):
        np.random.shuffle(training_data)
        batch = training_data[0:batch_size]
        if epoch == 0:
            # Initial error prior to training
            result = classify(batch, num_hidden_layers, num_neurons, weights)
            print "Initial training error:", result
            training_error.append(result)


        print "Epoch", epoch+1
        for data_idx in range(len(batch)):
            data_pt = batch[data_idx]
            # Iterating through the data points and computing the output of each layer
            for layer in range(num_hidden_layers + 1):
                weight_matrix = weights[layer]
                activation_results = []
                summation_results = []
                for neuron_idx in range(num_neurons[layer]):
                    weighted_sum = 0
                    if layer == 0:  # Initial layer
                        # Summation -> weights * input + bias
                        for idx in range(len(data_pt)-1):
                            weighted_sum += weight_matrix[neuron_idx][idx] * data_pt[idx]
                    else:
                        # Summation -> weights * input + bias
                        for idx in range(num_neurons[layer - 1]):
                            weighted_sum += weight_matrix[neuron_idx][idx] * activation[layer-1][idx]
                    weighted_sum += weight_matrix[neuron_idx][idx+1]  # bias
                    # Activation function -> Sigmoid
                    summation_results.append(weighted_sum)
                    activation_results.append(sigmoid_activation(weighted_sum))
                activation[layer] = activation_results
                summation[layer] = summation_results

            # Updating the weights - Back Propagation
            for layer in range(num_hidden_layers, -1, -1):
                is_output_layer = False
                activation_results = activation[layer]
                summation_results = summation[layer]
                delta_values = []
                if layer == num_hidden_layers:
                    is_output_layer = True
                else:
                    outgoing_weight_matrix = weights[layer+1]
                for neuron_idx in range(num_neurons[layer]):
                    delta_temp = 0
                    error = 0
                    if is_output_layer:
                        if activation_results[neuron_idx] == 1:
                            if neuron_idx != int(data_pt[len(data_pt)-1]):
                                error = -1
                        else:
                            if neuron_idx == int(data_pt[len(data_pt)-1]):
                                error = 1
                        delta_temp += error * gradient_sigmoid(summation_results[neuron_idx])
                    else:
                        for idx in range(num_neurons[layer+1]):
                            delta_temp += outgoing_weight_matrix[idx][neuron_idx]*delta[layer+1][idx]
                        delta_temp *= gradient_sigmoid(summation_results[neuron_idx])
                    delta_values.append(delta_temp)
                delta[layer] = delta_values

            for layer in range(num_hidden_layers, -1, -1):
                delta_values = delta[layer]
                if layer is not 0:
                    prev_activation_results = activation[layer-1]
                for neuron_idx in range(num_neurons[layer]):
                    # Update weights when delta is not zero
                    delta_value = delta_values[neuron_idx]
                    if delta_value is not 0:
                        if layer == 0:  # Initial layer
                            for idx in range(len(data_pt)-1):
                                change = learning_rate*delta_value*data_pt[idx] + \
                                         momentum*changes[layer][neuron_idx][idx]
                                weights[layer][neuron_idx][idx] += change
                                changes[layer][neuron_idx][idx] = change
                        else:  # The other layers
                            for idx in range(num_neurons[layer-1]):
                                change = learning_rate*delta_value*prev_activation_results[idx] + \
                                         momentum*changes[layer][neuron_idx][idx]
                                weights[layer][neuron_idx][idx] += change
                                changes[layer][neuron_idx][idx] = change
                        change = learning_rate*delta_value + momentum*changes[layer][neuron_idx][idx+1]
                        weights[layer][neuron_idx][idx+1] += change  # Updating bias
                        changes[layer][neuron_idx][idx+1] = change
        if (epoch + 1) % 10 == 0:
            result = classify(batch, num_hidden_layers, num_neurons, weights)
            print "Training error at epoch " + str(epoch+1) + ":", result
            training_error.append(result)

    print "Training complete..."
    print "Classifying test data and obtaining results..."
    return training_error, classify(test_data, num_hidden_layers, num_neurons, weights)


# Function to classify the data points and evaluate the model.
def classify(data, num_hidden_layers, num_neurons, weights):
    num_incorrect_classifications = 0
    # predicted_class_labels = []

    activation = {}
    for data_idx in range(len(data)):
        data_pt = data[data_idx]
        # Iterating through the data points and computing the output of each layer
        output_layer = num_hidden_layers
        for layer in range(num_hidden_layers + 1):
            weight_matrix = weights[layer]
            activation_results = []
            for neuron_idx in range(num_neurons[layer]):
                weighted_sum = 0
                if layer == 0:  # Initial layer
                    # Summation -> weights * input + bias
                    for idx in range(len(data_pt)-1):
                        weighted_sum += weight_matrix[neuron_idx][idx] * data_pt[idx]
                else:
                    # Summation -> weights * input + bias
                    for idx in range(num_neurons[layer - 1]):
                        weighted_sum += weight_matrix[neuron_idx][idx] * activation[layer-1][idx]
                weighted_sum += weight_matrix[neuron_idx][idx+1]  # bias
                if layer == output_layer:
                    activation_results.append(sigmoid(weighted_sum))
                else:
                    # Activation function with threshold-> Sigmoid
                    activation_results.append(sigmoid_activation(weighted_sum))
            activation[layer] = activation_results

        # soft-max approach for extracting the predicted class
        predicted_class = activation[output_layer].index(max(activation[output_layer]))
        # predicted_class_labels.append(predicted_class)
        # print activation[output_layer]
        # print "predicted class:" + str(predicted_class) + "\tclass label:" + str(int(data_pt[len(data_pt)-1]))
        if predicted_class != int(data_pt[len(data_pt)-1]):
            num_incorrect_classifications += 1

    print "Number of incorrect classifications:", num_incorrect_classifications
    print "Error rate:", num_incorrect_classifications/len(data)*100
    return num_incorrect_classifications/len(data)*100


