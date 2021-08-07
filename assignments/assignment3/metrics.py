import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # implement metrics!
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

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):

    true_positives = np.count_nonzero(prediction == ground_truth)
    accuracy = true_positives/len(prediction)

    return accuracy
