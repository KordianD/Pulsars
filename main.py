# We will perform machine learning on pulsars data

from SVM_Classification import SVM
import data
from KNN_Classification import KNN
from Log_Regression import Log_Regression

train_data, test_data, train_labels, test_labels = data.get_pulsar_data()


kernels_for_SVM = ['linear', 'poly', 'rbf', 'sigmoid']
gammas_for_SVM = [0.000001, 0.0001, 0.001]
coefs_for_SVM = [-1.0, 1.0, 0.0]
SVM_classification = SVM(train_data, test_data, train_labels, test_labels,
                         kernels_for_SVM,
                         gammas_for_SVM,
                         coefs_for_SVM)
SVM_classification.perform()
SVM_classification.get_sensitivity_and_precision()
SVM_classification.get_general_result()



distance = ['euclidean', 'cityblock', 'minkowski']
n_neighbours = [1, 3, 5, 7]
weights = ['uniform', 'distance']
KNN_classification = KNN(train_data, test_data, train_labels, test_labels,
                         distance, n_neighbours, weights)
KNN_classification.perform()


LR_classification = Log_Regression(train_data, test_data, train_labels, test_labels)
LR_classification.perform()







