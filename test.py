from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from adaboost import AdaBoost

# load and prepare data
data = load_breast_cancer()
X = data.data
y = data.target
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train the library classifiers
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_1 = adaboost.predict(X_test)


# train our implementation
adaboost_imp = AdaBoost(n_classifiers=4)
adaboost_imp.fit(X_train, y_train)
y_pred_2 = adaboost_imp.predict(X_test)


# check accuracy for both implementations
acc = accuracy_score(y_test, y_pred_1)
print("Accuracy (library): ", acc)

acc2 = accuracy_score(y_test, y_pred_2)
print("Our implementation: ", acc2)
