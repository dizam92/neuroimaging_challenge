# -*- coding: utf-8 -*-
__author__ = 'maoss2'

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
import os
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# ******************************************** Global Values Section ***************************************************
logging.basicConfig(level=logging.INFO)
train_path = 'datasets/train.csv'
test_path = 'datasets/test.csv'
param_Max_Depth = np.arange(1, 15, 2)
param_Min_Samples_Split = np.arange(2, 8)
param_Criterion = ["gini", "entropy"]
logger = logging.getLogger(__name__)
n_cv = 5
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
        cls_indexes = []
        d = pd.read_csv(path_file)
        d['Diagnosis'].replace(to_replace={'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3}, inplace=True)
        y = d['Diagnosis'].values
        # Stats section
        nb_classes = np.unique(y)
        print 'il y a {} classes'.format(nb_classes)
        for cls in nb_classes:
            cls_indexes.append(d.loc[d.Diagnosis == cls].index.values)
            print 'il y a {} patients dans la classe {}'.format(d.loc[d.Diagnosis == cls].values.shape[0], cls)
            # commentaires: il y a le meme nombre pour chaque classe! Interessant. On peut faire de la classif binaire
            # par catégorie en mode one_vs_one/ puis faire du one_vs_all
        # replace the gender by binary class
        d['GENDER'].replace(to_replace={'Female': 1, 'Male': 0}, inplace=True)
        # delete unnecessary columns
        d.drop(dropped_labels, axis=1, inplace=True)
        # features
        features_names = d.columns.values
        # build the dataset
        # f = h5py.File('dataset_complet', 'w')
        # f.create_dataset('data', data=d.values)
        # f.create_dataset('target', data=y)
        # f.create_dataset('features', data=features_names)
        return d.values, y, features_names
    else:
        dropped_labels = ['SUB_ID']
        d = pd.read_csv(path_file)
        # replace the gender by binary class
        d['GENDER'].replace(to_replace={'Female': 1, 'Male': 0}, inplace=True)
        # delete unnecessary columns
        d.drop(dropped_labels, axis=1, inplace=True)
        # features
        features_names = d.columns.values
        # build the dataset
        # f = h5py.File('dataset_complet', 'w')
        # f.create_dataset('data', data=d.values)
        # f.create_dataset('features', data=features_names)
        return d.values, features_names

if __name__ == '__main__':
    train_data, label_train, features_names = loader(path_file=train_path)
    test_data, _ = loader(path_file=test_path)
    logger.info('Decisions Trees section')
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
    pipe = Pipeline([('scaling', StandardScaler()), ('DT', DecisionTreeClassifier())])
    params = {'DT__criterion': param_Criterion,
              'DT__max_depth': param_Max_Depth,
              'DT__min_samples_split': param_Min_Samples_Split}
    clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
    clf.fit(x_train, y_train)
    print clf.best_estimator_
    print clf.best_params_
    pred = clf.predict(x_test)
    print {"Metrics": get_metrics(y_test, pred)}
    print
    # Print the feature ranking
    print("Feature ranking:")
    importances = clf.best_estimator_.named_steps['DT'].feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(5):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))

    dot_data = export_graphviz(clf.best_estimator_.named_steps['DT'], out_file=None,
                               feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
                               filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data)
    graph.write_pdf("mci.pdf")
