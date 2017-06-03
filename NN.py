import tensorflow as tf
from Statistics import get_performance


class NN(object):

    def __init__(self, train_data, test_data, train_labels, test_labels, layers, hidden_units, classes):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.layers = layers
        self.hidden_units = hidden_units
        self.classes = classes
        self.feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(self.train_data[0]))]
        self.sensitivity = []
        self.specificity = []
        self.precision = []
        self.general_result = []

    def get_train_inputs(self):
        x = tf.constant(self.train_data)
        y = tf.constant(self.train_labels)

        return x, y

    def get_test_inputs(self):
        x = tf.constant(self.test_data)
        y = tf.constant(self.test_labels)

        return x, y

    def get_sensitivity(self):
        return self.sensitivity

    def get_precision(self):
        return self.precision

    def get_specificity(self):
        return self.specificity

    def get_general_results(self):
        return self.general_result

    def perform(self):
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(self.train_data[0]))]

        for lay in self.layers:
            for hid in self.hidden_units:
                classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                            hidden_units=[hid] * lay,
                                                            n_classes=2)
                classifier.fit(input_fn=self.get_train_inputs, steps=5000)

                res = classifier.predict(input_fn=self.get_test_inputs)

                performance = get_performance(self.test_labels, res)
                self.sensitivity.append(performance['sensitivity'])
                self.specificity.append(performance['specificity'])
                self.precision.append(performance['precision'])

                self.general_result.append(100 * classifier.evaluate(input_fn=self.get_test_inputs,
                                       steps=1)["accuracy"])




