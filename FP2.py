import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

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

clf = KNeighborsClassifier()
clf.fit(train_data, train_labels)
res = clf.predict(test_data)

counter = 0.0
for g, p in zip(test_labels, res):
    if g == p == 1:
        counter+=1




print('Accuracy: ' + str(100*clf.score(test_data, test_labels)) + ' %')
cc = sum(test_labels)
print('Recognized  ' + str(int(counter)) + ' from ' + str(int(cc)) + ' real pulsars (' + str(100*counter/cc) + '%)')


# plot mean of the integrated profile(x), mean of the DM-SNR curve(y) and labels(z)
x = np.array(data[:5000, 1].astype(float))
y = np.array(data[:5000, 5].astype(float))
z = np.array(data[:5000, 8].astype(float))
plt.scatter(x, y, c=z)
#plt.show()