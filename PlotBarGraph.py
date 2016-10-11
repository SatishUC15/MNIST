import numpy as np
import matplotlib.pyplot as plt


def plot_bar_graphs(training_results, test_results, num_trials):

    ind = np.arange(num_trials)
    width = 0.35

    labels = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
    for idx1 in range(len(training_results[0])):
        training = []
        test = []
        for idx2 in range(num_trials):
            training.append(training_results[idx2][idx1])
            test.append(test_results[idx2][idx1])

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, training, width, color='b')

        rects2 = ax.bar(ind + width, test, width, color='k')

        ax.set_ylabel(labels[idx1])
        ax.set_title('Performance on Individual Trials - '+labels[idx1])
        ax.legend((rects1[0], rects2[0]), ('Training data', 'Test data'))

        plt.show()
