import logging
from sklearn.linear_model import LogisticRegression
from Statistics import get_performance

class Log_Regression(object):
    def __init__(self, train_data, test_data, train_labels, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.sensitivity = 0
        self.specificity = 0
        self.precision = 0
        self.general_result = 0


    def perform(self):

        logging.info('Computing Logistic Regression')
        clf = LogisticRegression()
        clf.fit(self.train_data, self.train_labels)
        res = clf.predict(self.test_data)

        performance = get_performance(self.test_labels, res)
        self.sensitivity = performance['sensitivity']
        self.specificity = performance['specificity']
        self.precision = performance['precision']

        self.general_result = 100 * clf.score(self.test_data, self.test_labels)

    def get_sensitivity(self):
        return self.sensitivity

    def get_specificity(self):
        return self.specificity

    def get_precision(self):
        return self.precision

    def get_general_results(self):
        return self.general_result
