import logging
from sklearn.linear_model import LogisticRegression
from Statistics import Statistics

class Log_Regression(object):
    def __init__(self, train_data, test_data, train_labels, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.sensitivity = 0
        self.specificity = 0
        self.general_result = 0
        self.statistics = Statistics()

    def perform(self):

        logging.info('Computing Logistic Regression')
        clf = LogisticRegression()
        clf.fit(self.train_data, self.train_labels)
        res = clf.predict(self.test_data)

        self.statistics.perform(self.test_labels, res)
        self.sensitivity = self.statistics.get_sensitivity()
        self.specificity = self.statistics.get_specificity()

        self.general_result = 100 * clf.score(self.test_data, self.test_labels)

    def get_sensitivity(self):
        return self.sensitivity

    def get_specificity(self):
        return self.specificity

    def get_general_results(self):
        return self.general_result
