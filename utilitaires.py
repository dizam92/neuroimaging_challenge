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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC


# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import SGD, Adam
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils

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

def pre_traitement():
    """ Pré traitement des données """
    # 1ere idée: Calculer les valeurs, mean, std, max, min, mediane, mode
    # 2eme idée: faire une PCA
    # 3eme idée: faire du clustering
    # 4eme idée: La normalisation est déja prise en compte par le standardScaler
    # 5eme idée:
    train_data, label_train, features_names = loader(path_file=train_path)
    test_data, _ = loader(path_file=test_path)
    # print features_names
    # print train_data
    d = pd.read_csv(train_path)
    print d
    # do it for each classe
    mmse_train_AD = train_data.loc[train_data.Diagnosis == 'AD'].MMSE_bl.values


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
    test_data, test_ids, _ = loader(path_file=test_path)
    logger.info('Decisions Trees section')
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
    # pipe = Pipeline([('scaling', StandardScaler()), ('DT', DecisionTreeClassifier())])
    pipe = Pipeline([('scaling', MinMaxScaler()), ('DT', DecisionTreeClassifier())])
    params = {'DT__criterion': param_Criterion,
              'DT__max_depth': param_Max_Depth,
              'DT__min_samples_split': param_Min_Samples_Split}

    clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
    clf.fit(x_train, y_train)
    # clf.fit(train_data, label_train)
    print clf.best_estimator_
    print clf.best_params_
    pred = clf.predict(train_data)
    print {"Train Metrics": get_metrics(label_train, pred)}
    pred = clf.predict(x_test)
    print {"Test Metrics": get_metrics(y_test, pred)}
    cnf_matrix = confusion_matrix(y_test, pred)
    print cnf_matrix
    print
    print("Feature ranking:")
    importances = clf.best_estimator_.named_steps['DT'].feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(10):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))

    # dot_data = export_graphviz(clf.best_estimator_.named_steps['DT'], out_file=None,
    #                            feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
    #                            filled=True, rounded=True, special_characters=True)
    # graph = pydot.graph_from_dot_data(dot_data)
    # graph.write_pdf("mci.pdf")

def final_submission():
    train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
    test_data, test_ids, _, y_true = loader(path_file=test_path)
    logger.info('Decisions Trees section')
    pipe = Pipeline([('scaling', MinMaxScaler()), ('DT', DecisionTreeClassifier(criterion='entropy',
                                                                                max_depth=5,
                                                                                min_samples_split=9))])
    pipe.fit(train_data, label_train)
    # construire l'arbre de décision ici:
    print("Feature ranking:")
    importances = pipe.named_steps['DT'].feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))

    dot_data = export_graphviz(pipe.named_steps['DT'], out_file=None,
                               feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
                               filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data)
    graph.write_pdf("mci.pdf")

    train_pred = pipe.predict(train_data)
    print {"Train Metrics": get_metrics(label_train, train_pred)}
    pred = pipe.predict(test_data.values)
    print {"Real Test Metrics": get_metrics(y_true, pred)}
    cnf_matrix = confusion_matrix(y_true, pred)
    print cnf_matrix
    # # 'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3
    # with open('sample_submission.csv', 'w') as f:
    #     f.write('SUB_ID,Diagnosis\n')
    #     for i, el in enumerate(pred):
    #         if el == 0:
    #             f.write('{},HC\n'.format(test_ids[i]))
    #         if el == 1:
    #             f.write('{},AD\n'.format(test_ids[i]))
    #         if el == 2:
    #             f.write('{},MCI\n'.format(test_ids[i]))
    #         if el == 3:
    #             f.write('{},cMCI\n'.format(test_ids[i]))

# def random_forest_experiences():
#     train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
#     test_data, test_ids, _, y_true = loader(path_file=test_path)
#     logger.info('Decisions Trees section')
#     pipe = Pipeline([('scaling', MinMaxScaler()), ('DT', RandomForestClassifier(n_estimators=10, random_state=42))])
#     params = {'DT__criterion': param_Criterion,
#                   'DT__max_depth': param_Max_Depth,
#                   'DT__min_samples_split': param_Min_Samples_Split,
#                   'DT__n_estimators': [1000, 2000]}
#     clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
#
#     clf.fit(train_data, label_train)
#     # construire l'arbre de décision ici:
#     print("Feature ranking:")
#     importances = clf.best_estimator_.named_steps['DT'].feature_importances_
#     indices = np.argsort(importances)[::-1]
#     for f in range(20):
#         print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features_names[indices[f]]))
#
#     # dot_data = export_graphviz(clf.best_estimator_.named_steps['DT'], out_file=None,
#     #                            feature_names=features_names, class_names=['HC', 'AD', 'MCI', 'cMCI'],
#     #                            filled=True, rounded=True, special_characters=True)
#     # graph = pydot.graph_from_dot_data(dot_data)
#     # graph.write_pdf("mci.pdf")
#     print clf.best_estimator_
#     train_pred = clf.predict(train_data)
#     print {"Train Metrics": get_metrics(label_train, train_pred)}
#     pred = clf.predict(test_data.values)
#     print {"Real Test Metrics": get_metrics(y_true, pred)}
#     cnf_matrix = confusion_matrix(y_true, pred)
#     print cnf_matrix
    # # 'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3
    # with open('sample_submission.csv', 'w') as f:
    #     f.write('SUB_ID,Diagnosis\n')
    #     for i, el in enumerate(pred):
    #         if el == 0:
    #             f.write('{},HC\n'.format(test_ids[i]))
    #         if el == 1:
    #             f.write('{},AD\n'.format(test_ids[i]))
    #         if el == 2:
    #             f.write('{},MCI\n'.format(test_ids[i]))
    #         if el == 3:
    #             f.write('{},cMCI\n'.format(test_ids[i]))


# def svm_multiclasses_experiences_rbf():
#     train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
#     test_data_original, test_data, _ = loader(path_file=test_path)
#     logger.info('SVM multiclasses section')
#     x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
#     pipe = Pipeline([('scaling', MinMaxScaler()), ('SVM', SVC(kernel='rbf', random_state=42))])
#     params = {'SVM__C': param_C,
#               'SVM__gamma': param_Gamma}
#     clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
#     clf.fit(x_train, y_train)
#     print clf.best_estimator_
#     print clf.best_params_
#     pred = clf.predict(train_data)
#     print {"Train Metrics": get_metrics(label_train, pred)}
#     pred = clf.predict(x_test)
#     print {"Test Metrics": get_metrics(y_test, pred)}
#     cnf_matrix = confusion_matrix(y_test, pred)
#     print cnf_matrix
#
# def svm_multiclasses_experiences_poly():
#     train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
#     test_data_original, test_data, _ = loader(path_file=test_path)
#     logger.info('SVM multiclasses section')
#     x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
#     pipe = Pipeline([('scaling', MinMaxScaler()), ('SVM', SVC(kernel='poly', random_state=42))])
#     params = {'SVM__C': param_C,
#               'SVM__degree': param_Degree}
#     clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
#     clf.fit(x_train, y_train)
#     print clf.best_estimator_
#     print clf.best_params_
#     pred = clf.predict(train_data)
#     print {"Train Metrics": get_metrics(label_train, pred)}
#     pred = clf.predict(x_test)
#     print {"Test Metrics": get_metrics(y_test, pred)}
#     cnf_matrix = confusion_matrix(y_test, pred)
#     print cnf_matrix
#
# def oneVSone():
#     train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
#     test_data, test_ids,_ = loader(path_file=test_path)
#     logger.info('OVo multiclasses section')
#     x_train, x_test, y_train, y_test = train_test_split(train_data, label_train, train_size=0.8, random_state=42)
#     pipe = Pipeline([('scaling', MinMaxScaler()), ('ovo', OneVsOneClassifier(estimator=LinearSVC(random_state=42, dual=False)))])
#     params = {'ovo__estimator__C': param_C}
#     clf = GridSearchCV(pipe, param_grid=params, cv=n_cv, n_jobs=n_jobs, verbose=1)
#     clf.fit(x_train, y_train)
#     print clf.best_estimator_
#     print clf.best_params_
#     pred = clf.predict(train_data)
#     print {"Train Metrics": get_metrics(label_train, pred)}
#     pred = clf.predict(x_test)
#     print {"Test Metrics": get_metrics(y_test, pred)}
#     cnf_matrix = confusion_matrix(y_test, pred)
#     print cnf_matrix
#
#
# def build_multiclass_model(input_dim, hidden_sizes, activation, lr, dropout, n_classes):
#     model = Sequential()
#     for i, hidden_size in enumerate(hidden_sizes):
#         layer_name = 'last_hidden' if i == len(hidden_sizes) - 1 else 'hidden_layer{}'.format(i + 1)
#         if i == 0:
#             model.add(Dense(hidden_size, input_dim=input_dim, activation=activation, name=layer_name))
#         else:
#             model.add(Dense(hidden_size, activation=activation, name=layer_name))
#             model.add(Dropout(dropout))
#     model.add(Dense(n_classes, activation='softmax'))
#     optimizer = Adam(lr=lr)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=optimizer,
#                   metrics=['accuracy'])
#     return model
#
# def neuronal_network():
#     # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     # ADAM = adam(lr=0.01, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#     np.random.seed(1255)
#     train_data_original, label_train_original, train_data, label_train, features_names = loader(path_file=train_path)
#     test_data, test_ids, _ = loader(path_file=test_path)
#     logger.info('Neural Network')
#     dummy_y_train = np_utils.to_categorical(label_train)
#     transformer = StandardScaler()
#     train_data = transformer.fit_transform(train_data)
#     x_train, x_test, y_train, y_test = train_test_split(train_data, dummy_y_train, train_size=0.8, random_state=42)
#     x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=42)
#     # layers_sizes = [[500, 100, 50], [300, 150, 25], [1000, 500], [750, 25]]
#     # activations = ['relu', 'tanh']
#     # learning_rates = [0.0001, 0.01, 0.1]
#     # droptouts = [0.25, 0.5, 0.75]
#     layers_sizes = [[500, 100, 50]]
#     activations = ['relu']
#     learning_rates = [0.001]
#     droptouts = [0.5]
#     params = list(ParameterGrid({'layers_sizes': layers_sizes, 'activations': activations,
#                                  'learning_rates': learning_rates, 'droptouts': droptouts}))
#     valid_scores = []
#     models_learned = []
#     for param in params:
#         model = build_multiclass_model(input_dim=429, hidden_sizes=param['layers_sizes'],
#                                        activation=param['activations'], lr=param['learning_rates'],
#                                        dropout=param['droptouts'], n_classes=4)
#         model.fit(x_train, y_train,
#                   nb_epoch=100,
#                   batch_size=5)
#         score = model.evaluate(x_valid, y_valid, batch_size=2)
#         print param
#         print score
#         valid_scores.append(score)
#         models_learned.append(model)
#     print valid_scores
#     best_model = np.argmax(valid_scores)
#     test_data = transformer.fit_transform(test_data.values)
#     pred = models_learned[best_model].predict(test_data, batch_size=test_data.shape[0])
#     pred = [np.argmax(el) for el in pred]
#     # 'HC': 0, 'AD': 1, 'MCI': 2, 'cMCI': 3
#     with open('sample_submission_2.csv', 'w') as f:
#         f.write('SUB_ID,Diagnosis\n')
#         for i, el in enumerate(pred):
#             if el == 0:
#                 f.write('{},HC\n'.format(test_ids[i]))
#             if el == 1:
#                 f.write('{},AD\n'.format(test_ids[i]))
#             if el == 2:
#                 f.write('{},MCI\n'.format(test_ids[i]))
#             if el == 3:
#                 f.write('{},cMCI\n'.format(test_ids[i]))


if __name__ == '__main__':
    # neuronal_network()
    # oneVSone()
    # svm_multiclasses_experiences_poly()
    # svm_multiclasses_experiences_rbf()
    # dt_experiences()
    # random_forest_experiences()
    final_submission()
