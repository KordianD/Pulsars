# This module will prepare data for our project
import numpy as np
import pandas

# loading and reshaping data
samples = 17898
data = pandas.read_csv("HTRU_2.csv", header=None)
data = np.array(data)

classes = np.array(data[:, 8].astype(float))
dataset = np.array(data[:, 1:7].astype(float))
train_data_length = int(0.7*samples)
test_data_length = samples - train_data_length

train_data = dataset[:train_data_length]
test_data = dataset[train_data_length:]
train_labels = classes[:train_data_length]
test_labels = classes[train_data_length:]


def get_pulsar_data():
    """ It returns train_data, test_data, train_labels, test_labels """
    return train_data, test_data, train_labels, test_labels

def get_number_of_samples():
    return len(data)