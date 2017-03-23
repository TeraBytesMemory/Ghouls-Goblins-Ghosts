#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.preprocessing import PolynomialFeatures


def encode_onehot(df, cols):
    """
    https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
    """
    vec_data = pd.get_dummies(df[cols], dummy_na=False)
    vec_data = vec_data.fillna(0.0).astype(int)
#    print(vec_data)
#    print(vec_data.dtypes)
    vec_data.columns = vec_data.columns.astype(object)
    vec_data.columns = [cols + '_' + str(v) for v in vec_data.columns]

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)

    return df

def process_features(train, test):
    tables = [test, train]

    print("Handling missing values...")

    print("Filling Nan...")
    numerical_features = test.select_dtypes(include=["float","int","bool"]).columns.values
    categorical_features = train.select_dtypes(include=["object"]).columns.values
    for table in tables:
        for feature in numerical_features:
            table[feature].fillna(train[feature].mean(),
                                  inplace=True) # replace by median value
#            if not feature == 'Id':
#                table[feature] = table[feature].map(lambda x: np.log(x + 1))
#            else:
#                print(table[feature])
        for feature in ['color']:
            table[feature].fillna('unknown',
                                  inplace=True) # replace by most frequent value

    print("Handling features...")
    for table in tables:
        table['bl_lt_0.2'] = table.bone_length.map(lambda x: x < 0.2)
        table['bl_mt_0.7'] = table.bone_length.map(lambda x: x > 0.7)
        table['hl_mt_0.7'] = table.hair_length.map(lambda x: x > 0.7)
#        table['bone_hair'] = table.bone_length * table.bone_length
#        table['bone_soul'] = table.bone_length * table.has_soul
#        table['bone_flesh'] = table.bone_length * table.rotting_flesh
#        table['fresh_hair'] = table.rotting_flesh * table.bone_length
#        table['fresh_soul'] = table.rotting_flesh * table.has_soul
#        table['hair_soul'] = table.hair_length * table.has_soul
#        table['rf_lt_0.2'] = table.rotting_flesh.map(lambda x: x < 0.2)
#        table['rf_mt_0.8'] = table.rotting_flesh.map(lambda x: x > 0.8)
#        table['hl_lt_0.2'] = table.hair_length.map(lambda x: x > 0.2)
#        table['hs_mt_0.7'] = table.has_soul.map(lambda x: x > 0.7)
#        table['hs_lt_0.2'] = table.has_soul.map(lambda x: x > 0.2)

    for feature in ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']:
        for table in tables:
            table[feature] = StandardScaler().fit_transform(table[feature])

    for table in tables:
        table['color'] = table.color.map({
            'white': 1,
            'black': 2,
            'clear': 3,
            'blue' : 4,
            'green': 5,
            'blood': 6,
            'unknown': 0
        })

    #table = encode_onehot(table, feature)
    train = encode_onehot(train, 'color')
    test = encode_onehot(test, 'color')


    return train, test


def data_format():

    train_df = pd.read_csv(os.path.abspath('../input/train.csv'))
    test_df = pd.read_csv(os.path.abspath('../input/test.csv'))

    train_df, test_df = process_features(train_df, test_df)

    #train_df = train_df.drop([], axis=1)
    #test_df = test_df.drop([], axis=1)

    train_x, train_y = train_data_format(train_df)
    test_x, test_ids = test_data_format(test_df)

    return train_x, train_y, test_x, test_ids


def data_format_df():
    train_df = pd.read_csv(os.path.abspath('../input/train.csv'))
    test_df = pd.read_csv(os.path.abspath('../input/test.csv'))

    train_df, test_df = process_features(train_df, test_df)

    return train_df, test_df


def train_data_format(df):

    y = df['type'].values

    df = df.drop(['id', 'type'], axis=1)

    kpca = KernelPCA(n_components=3, kernel='rbf', degree=2, gamma=0.1)
    transf = kpca.fit_transform(df.drop(['bl_lt_0.2',
                                         'bl_mt_0.7',
                                         #'hl_lt_0.2',
                                         'hl_mt_0.7',
#                                         'rf_lt_0.2',
#                                         'rf_mt_0.8',
#                                         'hs_mt_0.7',
#                                         'hs_lt_0.2'
    ], axis=1).values)

    x = df.values
    x = np.hstack((x, transf))

    print(df.dtypes)

    return x, y


def test_data_format(df):

    ids = df['id'].values
    df = df.drop(['id'], axis=1)

    kpca = KernelPCA(n_components=3, kernel='rbf', degree=2, gamma=0.1)
    transf = kpca.fit_transform(df.drop(['bl_lt_0.2',
                                         'bl_mt_0.7',
                                         #'hl_lt_0.2',
                                         'hl_mt_0.7',
#                                         'rf_lt_0.2',
#                                         'rf_mt_0.8',
#                                         'hs_mt_0.7',
#                                         'hs_lt_0.2'
    ], axis=1).values)

    x = df.values
    x = np.hstack((x, transf))

    return x, ids





