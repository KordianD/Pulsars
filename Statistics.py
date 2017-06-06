import numpy as np

def get_performance(labels, predictions):
    TP, FN, FP, TN = 0, 0, 0, 0
    for g, p in zip(labels, predictions):
        if g == p == 1:
            TP += 1
        if g == 1 and p == 0:
            FN += 1
        if g == 0 and p == 1:
            FP += 1
        if g == 0 and p == 0:
            TN += 1

    if TP or FN:
        sensitivity = TP / (TP + FN)
    else:
        sensitivity = 0

    if TN or FP:
        specificity = TN / (TN + FP)
    else:
        specificity = 0

    if TP or FP:
        precision = TP / (TP + FP)
    else:
        precision = 0

    return {'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity}


def fscore(precision, sensitivity):
    return 2 * (np.array(precision) * np.array(sensitivity)) / (np.array(precision) + np.array(sensitivity))

def roc(sensitivity, specificity):
    return np.array(sensitivity), (1 - np.array(specificity))

def output(sensitivity, specificity, precision):
    print('Sensitivity ' + str(sensitivity))
    print('Specificity ' + str(specificity))
    print('Precision ' + str(precision))
    print('F-score ' + str(fscore(precision, sensitivity)))
    print('ROC ' + str(roc(sensitivity, specificity)))
