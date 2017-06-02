# We will perform machine learning on pulsars data

from SVM_Classification import SVM
#import KNN_Classification
import data
#general_results, knn_sensitivity = KNN_Classification.get_results()

#print(general_results)
#print(knn_sensitivity)

train_data, test_data, train_labels, test_labels = data.get_pulsar_data()

kernels_for_SVM = ['linear', 'poly', 'rbf', 'sigmoid']
gammas_for_SVM = [0.000001, 0.0001, 0.001]
coefs_for_SVM = [-1.0, 1.0, 0.0]

SVM_classification = SVM(train_data, test_data, train_labels, test_labels,
                         kernels_for_SVM,
                         gammas_for_SVM,
                         coefs_for_SVM)

SVM_classification.perform()
SVM_classification.get_sensitivity()
SVM_classification.get_general_result()

