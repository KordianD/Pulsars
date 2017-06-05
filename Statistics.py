


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
        sensitivity = 100 * TP / (TP + FN)
    else:
        sensitivity = 0

    if TN or FP:
        specificity = 100 * TN / (TN + FP)
    else:
        specificity = 0

    if TP or FP:
        precision = 100 * TP / (TP + FP)
    else:
        precision = 0

    return {'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity}
