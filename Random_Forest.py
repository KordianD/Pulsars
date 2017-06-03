from sklearn.ensemble import RandomForestClassifier
import logging
from Statistics import get_performance



class RFC(object):

    def __init__(self, train_data, test_data, train_labels, test_labels, trees = [], depths = []):
        self.trees = trees
        self.depths = depths
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.sensitivity = []
        self.specificity = []
        self.general_result = []
        self.precision = []

    def perform(self):
        counter = 0
        for t in self.trees:
            for d in self.depths:
                counter += 1
                logging.info(' Computing RF ... '
                             + str(int(100 * counter / (len(self.trees) * len(self.depths))))
                             + '% done')

                clf = RandomForestClassifier(n_estimators=t, max_depth=d)
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
