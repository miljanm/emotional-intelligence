# import
import pickle as pk
import numpy as np

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# unpickle data
data_set = pk.load(open("../data/bracelet.pkl", "rb"))
X = data_set[:,0:-1]
np.random.shuffle(X)
y = data_set[:,-1].astype('int')-1


# Experiment with different classifiers
n_folds = 5

clf = LogisticRegression(dual=False, penalty='l1')
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("LogReg l1 regularized - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = LogisticRegression(dual=False, penalty='l2')
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("LogReg l2 regularized - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("KNN k=3 - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("KNN k=5 - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = KNeighborsClassifier(n_neighbors=7, weights='distance')
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("KNN k=7 - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = RandomForestClassifier(n_estimators=180,min_samples_split=4)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Random Forest - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = GradientBoostingClassifier(n_estimators=180,min_samples_split=4)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Gradient Boosting - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Linear SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(kernel='poly', degree=2, C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Polynomial (d=2) kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(kernel='poly', degree=3, C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Polynomial (d=3) kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC(kernel='sigmoid', C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=n_folds)
print("Sigmoid kernel SVM - Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
