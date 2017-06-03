
class Statistics(object):

    def perform(self, labels, predictions):
        self.TP, self.FN, self.FP, self.TN = 0, 0, 0, 0
        for g, p in zip(labels, predictions):
            if g == p == 1:
                self.TP += 1
            if g == 1 and p == 0:
                self.FN += 1
            if g == 0 and p == 1:
                self.FP += 1
            if g == 0 and p == 0:
                self.TN += 1

    def get_sensitivity(self):
        if self.TP + self.FN != 0:
            return 100 * self.TP / (self.TP + self.FN)
        else:
            return 0

    def get_specificity(self):
        if self.TN + self.FP != 0:
            return 100 * self.TN / (self.TN + self.FP)
        else:
            return 0

