import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # Implement computing accuracy
    true_positives = np.count_nonzero(prediction == ground_truth)
    accuracy = true_positives/len(prediction)

    return accuracy
