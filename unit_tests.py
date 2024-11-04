import numpy as np

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
        # minimum 90% accuracy, since we cannot check the exact outcomes
        self.assertGreater(accuracy, 0.9)