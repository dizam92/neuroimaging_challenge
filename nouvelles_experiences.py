# -*- coding: utf-8 -*-
__author__ = 'maoss2'

import logging
import numpy as np
import pandas as pd
import pydot
import os
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC

from copy import deepcopy
# ******************************************** Global Values Section ***************************************************
logging.basicConfig(level=logging.INFO)
train_path = 'datasets/train.csv'
test_path = 'datasets/test.csv'
param_Max_Depth = np.arange(1, 10, 1)
param_Min_Samples_Split = np.arange(2, 8)
param_Criterion = ["gini", "entropy"]
param_C = np.logspace(-6, 6, 20)
param_Gamma = np.logspace(-5, 2, 10)
param_Degree = [3, 4, 6, 7, 8]
logger = logging.getLogger(__name__)
n_cv = KFold(n_splits=5, random_state=42)
n_jobs = 8
# **********************************************************************************************************************
def get_metrics(y_test, predictions_binary):
    """Compute the metrics for classifiers predictors
    Params: y_test: real labels
            predictions_binary: the predicted labels
    Return: metrics: a dictionnary of the metrics"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)
    # print "True values are", y_test
    # print "Predictions values are", predictions_binary
    print "Metrics:"
    print "Accuracy: %.3f" % accuracy_score(y_test, predictions_binary)
    print "F1 Score: %.3f" % f1_score(y_test, predictions_binary, average='macro')
    print "Precision Score: %.3f" % precision_score(y_test, predictions_binary, average='macro')
    print "Recall Score: %.3f" % recall_score(y_test, predictions_binary, average='macro')
    # print "AUC: %.3f" % roc_auc_score(y_test, predictions_binary)

    metrics = {"accuracy": accuracy_score(y_test, predictions_binary),
               "f1_score": f1_score(y_test, predictions_binary, average='macro'),
               "precision": precision_score(y_test, predictions_binary, average='macro'),
               "recall": recall_score(y_test, predictions_binary, average='macro')
                # ,"auc": roc_auc_score(y_test, predictions_binary)
               }
    return metrics


def loader(path_file):
    """ Load the dataset with pandas
    :param path_file: the path of the file
    :return pandas dataframe of the data"""

    if path_file.find('train') != -1:
        dropped_labels = ['SUB_ID', 'Diagnosis']
        d = pd.read_csv(path_file)
        d['Diagnosis'].replace(to_replace={'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3}, inplace=True)
        y_without_outliers = deepcopy(d['Diagnosis'])
        y = d['Diagnosis'].values
        # Stats section
        nb_classes = np.unique(y)
        print 'il y a {} classes'.format(nb_classes)
        d['GENDER'].replace(to_replace={'Female': 1, 'Male': 0}, inplace=True)
        # delete unnecessary columns
        d.drop(dropped_labels, axis=1, inplace=True)
        # features
        features_names = d.columns.values
        d_without_outliers = deepcopy(d.drop([17, 152, 203], axis=0))
        y_without_outliers.drop([17, 152, 203], axis=0, inplace=True)
        return d.values, y, d_without_outliers.values, y_without_outliers.values, features_names
    else:
        # filtering only the real test set for confirmation how i perform on the real values
        new_df = pd.read_csv('/home/maoss2/Documents/Doctorat/Hiver2017/Article_relatif_Neuroscience/MLchallenge_dummy&test_ID.csv')

        # recuperer juste les indexs cibles
        dummies_test_values_indexes = new_df[new_df['row ID'].str.contains('DUM_')].index.values
        dropped_labels = ['SUB_ID']
        d = pd.read_csv(path_file)

        # replace the gender by binary class
        d['GENDER'].replace(to_replace={'Female': 1, 'Male': 0}, inplace=True)

        # store the real labels of test set
        temp_df = deepcopy(new_df[~new_df['row ID'].str.contains('DUM_')])
        temp_df['Diagnosis'].replace(to_replace={'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3}, inplace=True)
        y_true = temp_df['Diagnosis'].values

        # Newly add for the real test. Drop the index of dummies data
        d.drop(d.index[dummies_test_values_indexes], inplace=True)
        test_ids = d['SUB_ID'].values

        # delete unnecessary columns
        d.drop(dropped_labels, axis=1, inplace=True)

        # features
        features_names = d.columns.values
        return d, test_ids, features_names, y_true

def dt_experiences():
    train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
    test_data, test_ids, _, y_true = loader(path_file=test_path)
    logger.info('Decisions Trees section')
    # x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
    # pipe = Pipeline([('scaling', StandardScaler()), ('DT', DecisionTreeClassifier())])
    pipe = Pipeline([('scaling', MinMaxScaler()), ('DT', DecisionTreeClassifier(random_state=42))])
    params = {'DT__criterion': param_Criterion,
              'DT__max_depth': param_Max_Depth,
              'DT__min_samples_split': param_Min_Samples_Split}

    clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
    clf.fit(train_data, label_train)
    print clf.best_estimator_
    print clf.best_params_
    pred = clf.predict(train_data)
    print {"Train Metrics": get_metrics(label_train, pred)}
    pred = clf.predict(test_data.values)
    print {"Test Metrics": get_metrics(y_true, pred)}
    cnf_matrix = confusion_matrix(y_true, pred)
    print cnf_matrix
    print
    print("Feature ranking:")
    importances = clf.best_estimator_.named_steps['DT'].feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(10):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))

    dot_data = export_graphviz(clf.best_estimator_.named_steps['DT'], out_file=None,
                               feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
                               filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data)
    graph.write_pdf("new_mci_dt.pdf")


def oneVSone():
    train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
    test_data, test_ids, _, y_true = loader(path_file=test_path)
    logger.info('OVo multiclasses section')

    pipe = Pipeline([('scaling', MinMaxScaler()), ('ovo', OneVsOneClassifier(estimator=DecisionTreeClassifier()))])
    params = {'ovo__estimator__criterion': param_Criterion,
              'ovo__estimator__max_depth': param_Max_Depth,
              'ovo__estimator__min_samples_split': param_Min_Samples_Split}
    clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
    clf.fit(train_data, label_train)
    print clf.best_estimator_
    print clf.best_params_
    pred = clf.predict(train_data)
    print {"Train Metrics": get_metrics(label_train, pred)}
    pred = clf.predict(test_data.values)
    print {"Test Metrics": get_metrics(y_true, pred)}
    cnf_matrix = confusion_matrix(y_true, pred)
    print cnf_matrix
    print

def fourVSall():
    train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
    test_data, test_ids, _, y_true = loader(path_file=test_path)
    label_train = np.asarray([-1 if el == 3 else 1 for el in label_train])
    y_true = np.asarray([-1 if el == 3 else 1 for el in y_true])
    logger.info('Decisions Trees section')
    # x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
    # pipe = Pipeline([('scaling', StandardScaler()), ('DT', DecisionTreeClassifier())])
    pipe = Pipeline([('scaling', MinMaxScaler()), ('DT', DecisionTreeClassifier(random_state=42))])
    params = {'DT__criterion': param_Criterion,
              'DT__max_depth': param_Max_Depth,
              'DT__min_samples_split': param_Min_Samples_Split}

    clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
    clf.fit(train_data, label_train)
    print clf.best_estimator_
    print clf.best_params_
    pred = clf.predict(train_data)
    print {"Train Metrics": get_metrics(label_train, pred)}
    pred = clf.predict(test_data.values)
    print {"Test Metrics": get_metrics(y_true, pred)}
    cnf_matrix = confusion_matrix(y_true, pred)
    print cnf_matrix
    print
    print("Feature ranking:")
    importances = clf.best_estimator_.named_steps['DT'].feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(10):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))

    dot_data = export_graphviz(clf.best_estimator_.named_steps['DT'], out_file=None,
                               feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
                               filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data)
    graph.write_pdf("new_mci_dt_four_vs_all.pdf")

if __name__ == '__main__':
    dt_experiences()
