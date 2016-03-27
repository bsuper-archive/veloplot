import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
import random
import featurizer
import operator
import skflow
import tensorflow as tf
import sys
import utils

DATA_FOLDER = "data/"
CTL_FILES = DATA_FOLDER + "ctl*.csv" # No touch data
ACT_FILES = DATA_FOLDER + "act*.csv" # Touch data
CALIBRATION_FILE = "N_matrix_trial9.mat"

def preprocess(features, labels):
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    features, labels = shuffle(features, labels, \
                                random_state=random.randint(0, 1000))
    return features, labels, scaler

def get_preprocessed_train_data(ctl_files=CTL_FILES, act_files=ACT_FILES):
    features, labels = featurizer.get_feature_vector(ctl_files, act_files)
    features, labels, scaler = preprocess(features, labels)
    return features, labels, scaler

def get_test_data(test_file="test/sliding11.txt", calibration_file=CALIBRATION_FILE):
    df = utils.process_data_files(test_file, calibration_file)
    df_segs = featurizer.segment(df)
    test_data = np.array(map(lambda df_seg: featurizer.featurize(df_seg), df_segs))
    return test_data, df, df_segs

def clf_predict(clf, test_data, df=None, df_segs=None):
    X, Y, scaler = get_preprocessed_train_data()
    clf.fit(X, Y)
    test_data = scaler.transform(test_data)
    return clf.predict(test_data)

def clf_predict_and_visualize(clf, test_data, df, df_segs, \
                                columns=[["Fx", "Fy", "Fz"], "F_mag", ["Mx", "My", "Mz"], "M_mag", ["AX", "AY", "AZ"], "A_mag", ["GyroX", "GyroY", "GyroZ"], "Gyro_mag"], \
                                display=True, save_figure=False, output_dir="out/", output_filename="preds_vis.png"):
    preds = clf_predict(clf, test_data)
    color_intervals = []
    for i in xrange(len(df_segs)):
        if preds[i] == 1:
            color_intervals.append((min(df_segs[i]["time"]), max(df_segs[i]["time"])))
    utils.plot_columns(df, columns, display=display, save_figure=save_figure, output_dir=output_dir, output_filename=output_filename, color_intervals=color_intervals)
    return preds

#########################################
# RANDOM FORESTS
#########################################

def random_forests():
    return RandomForestClassifier(n_estimators=200, max_features='sqrt', oob_score=True)

def random_forests_cross_val(X, Y, k=10):
    clf = random_forests()
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores
    clf.fit(X, Y)
    print "OOB score:", clf.oob_score_
    sorted_feature_importances = sorted(zip(featurizer.get_feature_names(), clf.feature_importances_), \
                                    key=operator.itemgetter(1))
    print "Feature Importances:"
    print sorted_feature_importances
    return clf

def do_random_forests_cross_val():
    print "\nRunning Random Forests..."
    X, Y, scaler = get_preprocessed_train_data()
    clf = random_forests_cross_val(X, Y)
    return clf, scaler

#########################################
# GRADIENT BOOSTED TREES
#########################################

def xgb_trees():
    return GradientBoostingClassifier(n_estimators=200, max_features='sqrt')

def xgb_trees_cross_val(X, Y, k=10):
    clf = xgb_trees()
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores
    clf = clf.fit(X, Y)
    sorted_feature_importances = sorted(zip(featurizer.get_feature_names(), clf.feature_importances_), \
                                    key=operator.itemgetter(1))
    print "Feature Importances:"
    print sorted_feature_importances
    return clf

def do_xgb_trees_cross_val():
    print "\nRunning XGB Trees..."
    X, Y, scaler = get_preprocessed_train_data()
    clf = xgb_trees_cross_val(X, Y)
    return clf, scaler

#########################################
# SVM
#########################################

def svc():
    return SVC(probability=True)

def svc_cross_val(X, Y, k=10):
    clf = svc()
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    print "{0}-fold CV Acc Mean: ".format(k), cv_scores.mean(), "Scores: ", cv_scores
    clf = clf.fit(X,Y)
    return clf

def do_svc_cross_val():
    print "\nRunning SVC..."
    X, Y, scaler = get_preprocessed_train_data()
    clf = svc_cross_val(X, Y)
    return clf, scaler

#########################################
# NEURAL NETWORK
#########################################

def dnn(nn_lr=0.1, nn_steps=1000):
    def relu_dnn(X, y, hidden_units=[100, 100]):
        features = skflow.ops.dnn(X, hidden_units=hidden_units,
          activation=tf.nn.relu)
        return skflow.models.logistic_regression(features, y)
    clf = skflow.TensorFlowEstimator(model_fn=relu_dnn, n_classes=2,
        steps=nn_steps, learning_rate=nn_lr, batch_size=100)
    return clf

def dnn_cross_val(X, Y, k=10):
    clf = dnn()
    cv_scores = []
    for train_indices, test_indices in KFold(X.shape[0], n_folds=k, shuffle=True, random_state=random.randint(0, 1000)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        clf.fit(X_train, Y_train)
        score = metrics.accuracy_score(Y_test, clf.predict(X_test))
        cv_scores.append(score)
    print "{0}-fold CV Acc Mean: ".format(k), np.mean(cv_scores), "Scores: ", cv_scores
    return clf

def do_dnn_cross_val():
    print "\nRunning Neural Network..."
    X, Y, scaler = get_preprocessed_train_data()
    clf = dnn_cross_val(X, Y)
    return clf, scaler

#########################################
# ENSEMBLE CLASSIFIER
#########################################

def ensemble_clf():
    return VotingClassifier(estimators=[
       ('dnn', dnn()), ('rf', random_forests()), ('xgb', xgb_trees()), ('svm', svc())],
       voting='soft', weights=[1,1,1,1])

def ensemble_cross_val(X, Y, k=10):
    clf = ensemble_clf()
    cv_scores = []
    for train_indices, test_indices in KFold(X.shape[0], n_folds=k, shuffle=True, random_state=random.randint(0, 1000)):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        clf.fit(X_train, Y_train)
        score = metrics.accuracy_score(Y_test, clf.predict(X_test))
        cv_scores.append(score)
    print "{0}-fold CV Acc Mean: ".format(k), np.mean(cv_scores), "Scores: ", cv_scores
    return clf

def do_ensemble_cross_val():
    print "\nRunning Ensemble Cross Val..."
    X, Y, scaler = get_preprocessed_train_data()
    clf = ensemble_cross_val(X, Y)
    return clf, scaler

if __name__ == "__main__":
    do_random_forests_cross_val()
    do_xgb_trees_cross_val()
    do_svc_cross_val()
    do_dnn_cross_val()
    do_ensemble_cross_val()
