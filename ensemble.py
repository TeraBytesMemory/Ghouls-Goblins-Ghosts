#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
import numpy as np

from model import data_format
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.le = LabelEncoder()
        self.le.fit(['Ghost', 'Goblin', 'Ghoul'])

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            print (i, ":", "Fit ", clf)

            for j, (train_idx, test_idx) in enumerate(folds):

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                print(j)
                clf.fit(X_train, y_train)
                y_pred = self.le.transform(clf.predict(X_holdout)[:])

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = self.le.transform(clf.predict(T)[:])

            S_test[:, i] = S_test_i.mean(1)

        score = cross_val_score(self.stacker, S_train, y, cv=5)
        print('Score:', score)
        print('Mean:', np.average(score))

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]

        return y_pred



if __name__ == '__main__':
    X, y, test_data, ids = data_format()

    base_models = [
        SVC(kernel='linear'),
        XGBClassifier(n_estimators=1000, max_depth=5, missing=4.0, reg_alpha=4.0),
        RandomForestClassifier(n_estimators=1000, max_depth=5),
        RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'entropy', max_features = 'auto',
                               min_samples_split=5, min_weight_fraction_leaf=0.0,
                               max_leaf_nodes=60, max_depth=100),
        LogisticRegression(C=0.01, tol=0.0001, solver='newton-cg', multi_class='multinomial'),
        LogisticRegression(C=10.0, tol=0.0001, solver='newton-cg', multi_class='ovr'),
#        GradientBoostingClassifier(max_features='log2', n_estimators=1000, max_depth=40)
    ]
    stacker = MLPClassifier()

    ensemble = Ensemble(5, stacker, base_models)
    result = ensemble.fit_predict(X, y, test_data)
    result = result.astype(np.object)

    ids = ids.astype(int)
    output = np.hstack((ids[np.newaxis].T, result[np.newaxis].T))

    predictions_file = open("submission.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['id', 'type'])
    open_file_object.writerows(output)
    predictions_file.close()
    print('Done.')

