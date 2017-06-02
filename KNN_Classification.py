# This module will be operating on out data using KNN classifier
# We decided to use this classifier for 1, 3, 5, 7 neighbours
# with 2 diffrent metric : euclidean, cityblock, minkowski
# Also we will use 2 different "weights" functions : uniform and weights

import logging
from sklearn.neighbors import KNeighborsClassifier

class KNN(object):
    def __init__(self, train_data, test_data, train_labels, test_labels, distance =[], n_neighbours =[], weights=[]):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.distance = distance
        self.n_neighbours = n_neighbours
        self.weights = weights
        self.sensitivity = []
        self.general_result = []

    def perform(self):
        counter = 0
        for d in self.distance:
            for n in self.n_neighbours:
                for w in self.weights:
                    counter += 1
                    logging.info(' Computing KNN ... '
                                 + str(int(100*counter / (len(self.distance) * len(self.n_neighbours) * len(self.weights))))
                                 + '% done')
                    clf = KNeighborsClassifier(n_neighbors=n, metric=d, weights=w)
                    clf.fit(self.train_data, self.train_labels)
                    res = clf.predict(self.test_data)

                    TP = 0
                    FN = 0
                    for g, p in zip(self.test_labels, res):
                        if g == p == 1:
                            TP += 1
                        if g == 0 and p == 1:
                            FN += 1

                    self.sensitivity.append(100 * TP / (TP + FN))
                    self.general_result.append(100 * clf.score(self.test_data, self.test_labels))

    def get_sensitivity(self):
            return self.knn_sensitivity;

    def get_general_results(self):
            return self.general_result