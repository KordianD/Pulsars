# We will perform machine learning on pulsars data

from SVM_Classification import SVM
from Statistics import *
from KNN_Classification import KNN
from Log_Regression import Log_Regression
from NN import NN
from Random_Forest import RFC
import data
import pandas as pd


train_data, test_data, train_labels, test_labels = data.get_pulsar_data_reduced_to(5)


'''

kernels_for_SVM = ['linear', 'poly', 'sigmoid']
gammas_for_SVM = [0.000001]
coefs_for_SVM = [-1.0, 1.0, 0]
SVM_classification = SVM(train_data, test_data, train_labels, test_labels,
                         kernels_for_SVM,
                         gammas_for_SVM,
                         coefs_for_SVM)
SVM_classification.perform()

precision = SVM_classification.get_precision()
sensitivity = SVM_classification.get_sensitivity()
specificity = SVM_classification.get_specificity()

output(sensitivity, specificity, precision)
'''
'''
distance = ['euclidean']
n_neighbours = [1, 2, 3, 5]
weights = ['uniform', 'distance']
KNN_classification = KNN(train_data, test_data, train_labels, test_labels,
                         distance, n_neighbours, weights)
KNN_classification.perform()


precision = KNN_classification.get_precision()
sensitivity = KNN_classification.get_sensitivity()
specificity = KNN_classification.get_specificity()

output(sensitivity, specificity, precision)


LR_classification = Log_Regression(train_data, test_data, train_labels, test_labels)
LR_classification.perform()

precision = LR_classification.get_precision()
sensitivity = LR_classification.get_sensitivity()
specificity = LR_classification.get_specificity()

output(sensitivity, specificity, precision)


number_of_classes = 2
number_of_hidden_units = [2, 5, 10, 20]
number_of_layers = [1, 2, 5, 10]
NN_classification = NN(train_data, test_data, train_labels, test_labels,
                       number_of_layers, number_of_hidden_units, number_of_classes)
NN_classification.perform()

precision = NN_classification.get_precision()
sensitivity = NN_classification.get_sensitivity()
specificity = NN_classification.get_specificity()

output(sensitivity, specificity, precision)
'''


trees = [5, 10, 20]
depths = [10, 50, 100]
RFC_classification = RFC(train_data, test_data, train_labels, test_labels,
                         trees, depths)
RFC_classification.perform()

precision = RFC_classification.get_precision()
sensitivity = RFC_classification.get_sensitivity()
specificity = RFC_classification.get_specificity()

output(sensitivity, specificity, precision)
'''


data.show_data_in_2_dimensions(15000)




data = pd.DataFrame([
    ["SVM:"], SVM_classification.get_sensitivity(), SVM_classification.get_precision(), SVM_classification.get_specificity(),
    ["KNN:"], KNN_classification.get_sensitivity(), KNN_classification.get_precision(), KNN_classification.get_specificity(),
    ["NN:"], NN_classification.get_sensitivity(), NN_classification.get_precision(), NN_classification.get_specificity(),
    ["RF:"], RFC_classification.get_sensitivity(), RFC_classification.get_precision(), RFC_classification.get_specificity(),
    ["LR:"], [LR_classification.get_sensitivity()], [LR_classification.get_precision()], [LR_classification.get_specificity()]
        ])
data.to_csv("data.csv", index=False, header=False)

'''

