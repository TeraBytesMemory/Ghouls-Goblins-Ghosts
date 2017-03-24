#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
import numpy as np

from model import data_format
from hyperopt import fmin, tpe, hp, rand
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def hp_choice(params):
    return dict([(key, hp.choice(key, value))
                      for key, value in params.items()])


def estimator(CLF):
    def inner(args):
        print("Args:", args)
        clf = CLF(**args)

        cross_valid = cross_val_score(clf, X, y, cv=5)
        acu = np.average(cross_valid)

        #print("Accurate:", acu)
        print("score:", cross_valid)
        print("mean:", acu)
        return -acu
    return inner


if __name__ == '__main__':

    gbc_parameters = {
        'n_estimators': [500, 750, 1000],
        'max_depth': [5, 10, 15, 20, 30, 40],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    svc_parameters = {
        'C': [20.0, 50.0],
        'epsilon': [0.5, 1.0],
        'kernel': ['linear']
    }

    svc_parameters = {
        'C': [20.0, 50.0, 100.0],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }


    lr_parameters = {
        'solver' : ['newton-cg', 'lbfgs'],
        'multi_class' : ['ovr', 'multinomial'],
        'C' : [0.005, 0.01, 1, 10, 100, 1000],
        'tol': [0.0001, 0.001, 0.005]
    }

    xgb_parameters = {
        'n_estimators': [500, 750, 1000],
        'max_depth': [5, 10, 15, 20, 30, 40],
        'reg_alpha': [2.0, 3.0, 4.0],
        'missing': [np.nan, 1.0, 2.0, 3.0, 4.0]
    }

    rf_parameters = {
        'n_estimators': [250, 500, 750, 1000],
        'max_depth': [10, 15, 20],
    }

    mlp_parameters = {
        'hidden_layer_sizes': [(256,), (512,), (1024,)],
        'alpha': [1.0, 0.01, 0.001, 0.0001],
        'max_iter': [400, 800, 1600],
    }


    X, y, test_data, ids = data_format()

    best = fmin(estimator(LogisticRegression),
                hp_choice(lr_parameters),
                algo=tpe.suggest,
                max_evals=30)
    best = dict([(key, lr_parameters[key][value]) for key, value in best.items()])

    print("\nBest Model...")
    estimator(LogisticRegression)(best)

    clf = LogisticRegression(**best)
    clf.fit(X, y)

    result = clf.predict(test_data)
    result = result.astype(np.object)

    ids = ids.astype(int)
    output = np.hstack((ids[np.newaxis].T, result[np.newaxis].T))

    predictions_file = open("submission.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['id', 'type'])
    open_file_object.writerows(output)
    predictions_file.close()
    print('Done.')

