# This module will be operating on out data using KNN classifier
# We decided to use this classifier for 1, 3, 5, 7 neighbours
# with 2 diffrent metric : euclidean, cityblock, minkowski
# Also we will use 2 different "weights" functions : uniform and weights

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
import data

train_data, test_data, train_labels, test_labels = data.get_pulsar_data()

distance = ['euclidean', 'cityblock', 'minkowski']
n_neighbours = [1, 3, 5, 7]
weights = ['uniform', 'distance']

knn_sensitivity = []
general_result = []

for d in distance:
    for n in n_neighbours:
        for w in weights:

            clf = KNeighborsClassifier(n_neighbors=n, metric=d, weights=w)
            clf.fit(train_data, train_labels)
            res = clf.predict(test_data)

            TP = 0
            FN = 0
            for g, p in zip(test_labels, res):
                if g == p == 1:
                    TP += 1
                if g == 0 and p == 1:
                    FN += 1

            knn_sensitivity.append(100*TP/(TP + FN))
            general_result.append(100*clf.score(test_data, test_labels))


def get_results():
    " It returns general_data and sensitivity"
    return general_result, knn_sensitivity