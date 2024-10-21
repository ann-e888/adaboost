import numpy as np

class DecisionStump:
    def __init__(self):
        # the direction of classification for each sample
        self.polarity = 1
        # id of a column
        self.feature_idx = None
        # determines whether the value should be 1 or -1
        self.threshold = None
        # weight of the classifier
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)

        # determining the value of a prediction
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column >= self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_classifiers=5):
        # initialize the classification with a specific num of learners
        self.n_classifiers = n_classifiers
        self.clfs = []

    def fit(self, X, y):
        # initialize number of rows, and columns
        n_samples, n_features = X.shape
        # create a matrix of size n_samples, and assign the same weight to every sample
        weights = np.full(n_samples, (1 / n_samples)) 


        # create the classifiers
        for _ in range(self.n_classifiers):
            clf = DecisionStump()
            min_error = float('inf')

            # for each column in the data set calculate the thresholds
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                p = 1


                # calculate the number of miscalculated predctions
                for threshold in thresholds:
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = weights[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate the new weight (alpha) of the classifier
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
           
            # reassign predctions and weights
            predictions = clf.predict(X)
            weights *= np.exp(-clf.alpha * y * predictions)
            weights /= np.sum(weights)
            self.clfs.append(clf)


    # create the prediction for the dataset, scaling each classifier by its weight 
    def predict(self, X):
        return np.sign(np.sum([clf.alpha * clf.predict(X) for clf in self.clfs], axis=0))
