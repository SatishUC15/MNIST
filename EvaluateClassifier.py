# Function to compute and display the sensitivity, specificity, PPV and NPV of a classifier.
from __future__ import division


def evaluate_classifier(true_positive, true_negative, false_positive, false_negative):

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (false_positive + true_negative)
    PPV = true_positive / (true_positive + false_positive)
    NPV = true_negative / (true_negative + false_negative)

    return [sensitivity, specificity, PPV, NPV]
