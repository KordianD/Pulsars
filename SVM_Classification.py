from sklearn.svm import SVC
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class SVM(object):

    def __init__(self, train_data, test_data, train_labels, test_labels, kernels = [], gammas = [], coefs = []):
        self.kernels = kernels
        self.gammas = gammas
        self.coefs = coefs
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.sensitivity = []
        self.general_result = []
        self.precision = []

    def perform(self):
        counter = 0
        for k in self.kernels:
            for g in self.gammas:
                for c in self.coefs:
                    counter += 1
                    logging.info(' Computing SVM ... '
                                 + str(int(100 * counter / (len(self.coefs) * len(self.gammas) * len(self.kernels)
                                 - (self.kernels.count('linear') * len(self.gammas) * len(self.coefs) - 1)
                                 - (self.kernels.count('rbf') * len(self.gammas) * (len(self.coefs) - 1)))))
                                 + '% done')

                    clf = SVC(kernel=k, gamma=g, coef0=c)
                    clf.fit(self.train_data, self.train_labels)
                    res = clf.predict(self.test_data)

                    true_positives = 0
                    false_negatives = 0
                    false_positives = 0
                    for ideal, predicted in zip(self.test_labels, res):
                        if ideal == predicted == 1:
                            true_positives += 1
                        if ideal == 0 and predicted == 1:
                            false_negatives += 1
                        if ideal == 1 and predicted == 0:
                            false_positives += 1

                    if true_positives or false_negatives:
                        self.sensitivity.append(100 * true_positives / (true_positives + false_negatives))
                    else:
                        self.sensitivity.append(0)
                    if true_positives or false_negatives:
                        self.precision.append(100 * true_positives / (true_positives + false_positives))
                    else:
                        self.precision.append(0)
                    self.general_result.append(100 * clf.score(self.test_data, self.test_labels))

                    if k == 'linear' or k == 'rbf':
                        break
                if k == 'linear':
                    break

    def get_sensitivity_and_precision(self):
        counter = -1
        for k in self.kernels:
            for g in self.gammas:
                for c in self.coefs:
                    counter += 1
                    print('Kernel: ' + str(k) + ' Gamma: ' + str(g) + ' Coef0: ' + str(c)
                          + ' Sensitivity: ' + "%.2f" % self.sensitivity[counter] + "%"
                          + ' Precision: ' + "%.2f" % self.precision[counter] + '%')

                    if k == 'linear' or k == 'rbf':
                        break
                if k == 'linear':
                    break
        print("-" * 70)

    def get_general_result(self):
        counter = -1
        for k in self.kernels:
            for g in self.gammas:
                for c in self.coefs:
                    counter += 1
                    print('Kernel: ' + str(k) + ' Gamma: ' + str(g) + ' Coef0: ' + str(c)
                          + ' General Result: ' "%.2f" % self.general_result[counter] + '%')

                    if k == 'linear' or k == 'rbf':
                        break
                if k == 'linear':
                    break
        print("-" * 70)