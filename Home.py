from sklearn.feature_selection import VarianceThreshold
import numpy as np
import MLP_Backpropagation as mlp
import PlotTrainingError

# Importing the MNIST data set attributes
dataSet = np.genfromtxt('MNISTnumImages5000.txt')

# Importing corresponding MNIST class labels
classLabels = np.genfromtxt('MNISTnumLabels5000.txt')

data = np.concatenate((dataSet, np.array([classLabels]).T), axis=1)

num_hidden_layers = 1
num_hidden_neurons = [100]
num_output_neurons = 10
learning_rate = 0.5
momentum = 0.1

mlp_training_error, result = mlp.train_and_classify(data[0:3000], num_hidden_layers, num_hidden_neurons,
                                                    num_output_neurons, learning_rate, momentum)
PlotTrainingError.plot_training_error(mlp_training_error)
