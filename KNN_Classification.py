# This module will be operating on out data using KNN classifier
# We decided to use this classifier for 1, 3, 5, 7 neighbours
# with 2 diffrent metric : euclidean, cityblock, minkowski
# Also we will use 2 different "weights" functions : uniform and weights

import logging
from sklearn.neighbors import KNeighborsClassifier
from Statistics import get_performance

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
        self.specificity = []
        self.precision = []
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

                    performance = get_performance(self.test_labels, res)
                    self.sensitivity.append(performance['sensitivity'])
                    self.specificity.append(performance['specificity'])
                    self.precision.append(performance['precision'])

                    self.general_result = 100 * clf.score(self.test_data, self.test_labels)

    def get_sensitivity(self):
            return self.sensitivity

    def get_specificity(self):
        return self.specificity

    def get_precision(self):
        return self.precision

    def get_general_results(self):
        return self.general_result