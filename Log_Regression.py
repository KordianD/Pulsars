import logging
from sklearn.linear_model import LogisticRegression


class Log_Regression(object):
    def __init__(self, train_data, test_data, train_labels, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.sensitivity = []
        self.general_result = []

    def perform(self):

        logging.info('Computing Logistic Regression')
        clf = LogisticRegression()
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
            return self.sensitivity

    def get_general_results(self):
            return self.general_result
