import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # Implement metrics
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_positives, true_negatives, false_negatives, false_positives = 0, 0, 0, 0
    
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            true_positives += 1

        if not prediction[i] and not ground_truth[i]:
            true_negatives += 1

        if not prediction[i] and ground_truth[i]:
            false_negatives += 1

        if prediction[i] and not ground_truth[i]:
            false_positives += 1

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    f1 = 2/(recall**-1 + precision**-1)
    accuracy = np.count_nonzero(prediction == ground_truth)/len(prediction)

    return precision, recall, f1, accuracy

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # Implement computing accuracy

    true_positives = np.count_nonzero(prediction == ground_truth)
    accuracy = true_positives/len(prediction)

    return accuracy
