import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from unittest import TestCase

from adaboost import DecisionStump, AdaBoost


class TestDecisionStump(TestCase):
    
    def test_decision_stump_pos_polarity(self):
        stump = DecisionStump()
        stump.polarity = 1
        stump.threshold = 1.5
        stump.feature_idx = 0

        X = np.array([[1], [2], [1.2], [0.5], [1.7]])
        expected_predictions = np.array([-1, 1, -1, -1, 1])

        predictions = stump.predict(X)
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_decision_stump_neg_polarity(self):
        stump = DecisionStump()
        stump.polarity = -1
        stump.threshold = 1.5
        stump.feature_idx = 0

        X = np.array([[1], [2], [1.2], [0.5], [1.7]])
        expected_predictions = np.array([1, -1, 1, 1, -1])

        predictions = stump.predict(X)
        np.testing.assert_array_equal(predictions, expected_predictions)


class TestAdaBoost(TestCase):

    def test_fit_single_clf(self):
        X = np.array([[1], [2], [1.1], [2.1]])
        y = np.array([-1, 1, -1, 1])
        
        clf = AdaBoost(n_classifiers=1)
        clf.fit(X, y)

        self.assertEqual(len(clf.clfs), 1)
        
        stump = clf.clfs[0]
        self.assertIsNotNone(stump.feature_idx)
        self.assertIsNotNone(stump.threshold)
        self.assertIsNotNone(stump.alpha)


    def test_fit_multiple_clfs(self):
        X = np.array([[1], [2], [1.1], [2.1]])
        y = np.array([-1, 1, -1, 1])
        
        clf = AdaBoost(n_classifiers=3)
        clf.fit(X, y)

        self.assertEqual(len(clf.clfs), 3)
        
        for stump in clf.clfs:
            self.assertIsNotNone(stump.feature_idx)
            self.assertIsNotNone(stump.threshold)
            self.assertIsNotNone(stump.alpha)


    def test_predict_single_clf(self):
        X = np.array([[1], [2], [1.1], [2.1]])
        y = np.array([-1, 1, -1, 1])
        
        clf = AdaBoost(n_classifiers=1)
        clf.fit(X, y)
        predictions = clf.predict(X)

        np.testing.assert_array_equal(predictions, y)


    def test_predict_multiple_clfs_accuracy(self):
        X = np.array([[1], [2], [1.1], [2.1]])
        y = np.array([-1, 1, -1, 1])
        
        clf = AdaBoost(n_classifiers=4)
        clf.fit(X, y)
        predictions = clf.predict(X)

        accuracy = np.mean(predictions == y)
        self.assertGreater(accuracy, 0.8)


    def test_breast_cancer(self):
        X, y = load_breast_cancer(return_X_y=True)
        y = np.where(y == 0, -1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = AdaBoost(n_classifiers=4)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.assertGreater(accuracy, 0.8, "AdaBoost accuracy on Breast Cancer dataset should be above 80%")
        

    def test_digits(self):
        X, y = load_digits(return_X_y=True)
        y = np.where(y < 5, -1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = AdaBoost(n_classifiers=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.assertGreater(accuracy, 0.8, "AdaBoost accuracy on Digits dataset should be above 80%")


    def test_wine(self):
        X, y = load_wine(return_X_y=True)
        y = np.where(y < 1, -1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = AdaBoost(n_classifiers=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.assertGreater(accuracy, 0.8, "AdaBoost accuracy on Digits dataset should be above 80%")