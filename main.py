# We will perform machine learning on pulsars data

from SVM_Classification import SVM
from KNN_Classification import KNN
from Log_Regression import Log_Regression
from NN import NN
from Random_Forest import RFC
import data
import pandas as pd


train_data, test_data, train_labels, test_labels = data.get_pulsar_data()



kernels_for_SVM = ['linear', 'poly', 'rbf', 'sigmoid']
gammas_for_SVM = [0.000001, 0.0001, 0.001]
coefs_for_SVM = [-1.0, 1.0, 0.0]
SVM_classification = SVM(train_data, test_data, train_labels, test_labels,
                         kernels_for_SVM,
                         gammas_for_SVM,
                         coefs_for_SVM)
SVM_classification.perform()
print(SVM_classification.get_sensitivity())
print(SVM_classification.get_specificity())
print(SVM_classification.get_precision())



distance = ['euclidean', 'cityblock', 'minkowski']
n_neighbours = [1, 2, 3, 5]
weights = ['uniform', 'distance']
KNN_classification = KNN(train_data, test_data, train_labels, test_labels,
                         distance, n_neighbours, weights)
KNN_classification.perform()
print('\nKNN\n')
print(KNN_classification.get_general_results())
print('\n')
print(KNN_classification.get_sensitivity())



LR_classification = Log_Regression(train_data, test_data, train_labels, test_labels)
LR_classification.perform()
print('\nLN')
print(LR_classification.get_general_results())
print('\n')
print(LR_classification.get_sensitivity())
print('\n')



number_of_classes = 2
number_of_hidden_units = [2, 5, 10, 20]
number_of_layers = [1, 2, 5, 10]
NN_classification = NN(train_data, test_data, train_labels, test_labels,
                       number_of_layers, number_of_hidden_units, number_of_classes)
NN_classification.perform()
print('\n NETWORK \n')
print(NN_classification.get_general_results())
print(NN_classification.get_sensitivity())



trees = [5, 10, 20]
depths = [10, 50, 100]
RFC_classification = RFC(train_data, test_data, train_labels, test_labels,
                         trees, depths)
RFC_classification.perform()
print(RFC_classification.get_sensitivity())
print(RFC_classification.get_precision())
print(RFC_classification.get_specificity())



data = pd.DataFrame([
    ["SVM:"], SVM_classification.get_sensitivity(), SVM_classification.get_precision(), SVM_classification.get_specificity(),
    ["KNN:"], KNN_classification.get_sensitivity(), KNN_classification.get_precision(), KNN_classification.get_specificity(),
    ["NN:"], NN_classification.get_sensitivity(), NN_classification.get_precision(), NN_classification.get_specificity(),
    ["RF:"], RFC_classification.get_sensitivity(), RFC_classification.get_precision(), RFC_classification.get_specificity(),
    ["LR:"], [LR_classification.get_sensitivity()], [LR_classification.get_precision()], [LR_classification.get_specificity()]
        ])
data.to_csv("data.csv", index=False, header=False)



