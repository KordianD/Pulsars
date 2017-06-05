# This module will prepare data for our project
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# loading and reshaping data
samples = 17898
data = pandas.read_csv("HTRU_2.csv", header=None)
data = np.array(data)
np.random.shuffle(data)

classes = np.array(data[:, 8].astype(float))
dataset = np.array(data[:, 0:8].astype(float))
dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)


def get_pulsar_data():
    """ It returns train_data, test_data, train_labels, test_labels """
    train_data_length = int(0.7*samples)
    test_data_length = samples - train_data_length

    train_data = dataset[:train_data_length]
    test_data = dataset[train_data_length:]
    train_labels = classes[:train_data_length]
    test_labels = classes[train_data_length:]
    return train_data, test_data, train_labels, test_labels


def get_pulsar_data_reduced_to(n_features):
    """ It returns train_data, test_data, train_labels, test_labels reduced to n_features by PCA"""
    pca = PCA(n_components=n_features)
    datasetPCA = pca.fit_transform(dataset)
    train_data_length = int(0.7 * samples)
    test_data_length = samples - train_data_length

    train_data = datasetPCA[:train_data_length]
    test_data = datasetPCA[train_data_length:]
    train_labels = classes[:train_data_length]
    test_labels = classes[train_data_length:]
    return train_data, test_data, train_labels, test_labels


def get_number_of_samples():
    return len(data)


def show_data_in_2_dimensions(n_samples):
    pca = PCA(n_components=2)
    datasetPCA = pca.fit_transform(dataset)
    x = np.zeros(len(datasetPCA))
    y = np.zeros(len(datasetPCA))
    for i in range(len(datasetPCA)):
        x[i] = datasetPCA[i][0]
        y[i] = datasetPCA[i][1]
    plt.scatter(x[:n_samples], y[:n_samples], c=classes[:n_samples])
    plt.show()