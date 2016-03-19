import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import random
import featurizer
import operator
import skflow
import tensorflow as tf

DATA_FOLDER = "data/"
CTL_FILES = DATA_FOLDER + "ctl*.csv"
ACT_FILES = DATA_FOLDER + "act*.csv"

def preprocess(features, labels):
    features = preprocessing.scale(features)
    features, labels = shuffle(features, labels, \
                                random_state=random.randint(0, 1000))
    return features, labels

def get_XY():
    features, labels = featurizer.get_feature_vector(CTL_FILES, ACT_FILES)
    features, labels = preprocess(features, labels)
    return features, labels

def random_forests(X, Y, k=5):
    clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', oob_score=True)
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores
    clf.fit(X, Y)
    print "OOB score:", clf.oob_score_
    sorted_feature_importances = sorted(zip(featurizer.get_feature_names(), clf.feature_importances_), \
                                    key=operator.itemgetter(1))
    print "Feature Importances:"
    print sorted_feature_importances

def do_random_forests():
    print "\nRunning Random Forests..."
    X, Y = get_XY()
    random_forests(X, Y)

def xgb_trees(X, Y, k=5):
    clf = GradientBoostingClassifier(n_estimators=200, max_features='sqrt')
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores
    clf.fit(X, Y)
    sorted_feature_importances = sorted(zip(featurizer.get_feature_names(), clf.feature_importances_), \
                                    key=operator.itemgetter(1))
    print "Feature Importances:"
    print sorted_feature_importances

def do_xgb_trees():
    print "\nRunning XGB Trees..."
    X, Y = get_XY()
    xgb_trees(X, Y)

def svc(X, Y, k=10):
    clf = SVC()
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores

def do_svc():
    print "\nRunning SVC..."
    X, Y = get_XY()
    svc(X, Y)

def dnn(X, Y, k=10, nn_lr=0.1, nn_steps=1000):
    def relu_dnn(X, y, hidden_units=[100, 100]):
        features = skflow.ops.dnn(X, hidden_units=hidden_units,
          activation=tf.nn.relu)
        return skflow.models.logistic_regression(features, y)
    clf = skflow.TensorFlowEstimator(model_fn=relu_dnn, n_classes=2,
        steps=nn_steps, learning_rate=nn_lr, batch_size=100)
    cv_scores = []
    for train_indices, test_indices in KFold(X.shape[0], n_folds=k, shuffle=True, random_state=random.randint(0, 1000)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        clf.fit(X_train, Y_train)
        score = metrics.accuracy_score(Y_test, clf.predict(X_test))
        cv_scores.append(score)
    print "{0}-fold CV Acc Mean: ".format(k), np.mean(cv_scores), "Scores: ", cv_scores

def do_dnn():
    print "\nRunning Neural Network..."
    X, Y = get_XY()
    dnn(X, Y)

if __name__ == "__main__":
    do_random_forests()
    do_xgb_trees()
    do_svc()
    do_dnn()
