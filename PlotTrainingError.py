import matplotlib.pyplot as plt
import numpy as np


def plot_training_error(training_error):
    # Function to plot the training error and the mean training error associated with the Perceptron.

    x_values = np.array(range(0, 90, 10))
    # mean_training_error = training_error[0]
    # for idx1 in range(len(training_error[0])):
    #     mean_training_error[idx1] = float(training_error[0][idx1])
    # plt.figure(1)
    # for idx in range(num_trails):
    #     plt.plot(x_values, np.array(training_error[idx]))
    #     if idx > 0:
    #         for idx1 in range(len(training_error[idx])):
    #             mean_training_error[idx1] += float(training_error[idx][idx1])
    #     plt.autoscale(enable=True, axis=u'both', tight=False)
    # plt.xlabel('Epochs')
    # plt.ylabel('Training Error Rate')
    # plt.title('Trial-Wise Training Error For Perceptron')
    # plt.legend(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8', 'Trial 9'])

    # size = len(mean_training_error)
    # for idx1 in range(size):
    #     mean_training_error[idx1] /= size

    plt.figure(2)
    plt.plot(x_values, training_error)
    plt.autoscale(enable=True, axis=u'both', tight=False)
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.title('Time series of Training Error')
    plt.show()


