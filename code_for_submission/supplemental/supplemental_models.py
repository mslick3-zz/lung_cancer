import itertools
import sys
import os

import re
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,\
AdaBoostClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from sklearn.qda import QDA

def hyperTune(x_train, y_train):
    base_models = [
        QDA(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        AdaBoostClassifier()
    ]

    params = [
        {},
        {'n_estimators':[150,300], 'learning_rate':[.01,.1,.18], 'max_leaf_nodes':[None]},
        {'n_estimators':[200,300], 'max_features':['auto', 'log2'], 'class_weight':['balanced_subsample']},
        {'C':[.01,.1,10,100]},
        {'n_estimators':[100,200,300]},
    ]

    print("Starting optimization ...")
    
    best_models = []
    
    for i, model in enumerate(base_models):
        CV_rfc = GridSearchCV(estimator=model, param_grid=params[i], cv=5, scoring='recall', n_jobs=-1)
        CV_rfc.fit(x_train, y_train)
        best_models.append(CV_rfc.best_estimator_)
        
    print("Done optimizing model parameters!")
    
    return best_models
    
def voted(x_train, y_train, x_test, est_weights, models, prob=False):
    voting_models = [(re.sub(r'[^A-Z]', '', str(m))[0:2].lower(), m) for m in models]
    vc = VotingClassifier(estimators=voting_models, voting='soft', weights=est_weights)
    vc.fit(x_train, y_train)
    if prob:
        Y_pred_voted = vc.predict_proba(x_test)[:,1]
    else:
        Y_pred_voted = vc.predict(x_test)
    return Y_pred_voted

def scaledPolyFeatures(x_train, x_test, columns):
    x_train = MaxAbsScaler().fit_transform(x_train)
    x_test = MaxAbsScaler().fit_transform(x_test)

    poly = PolynomialFeatures(degree=2, interaction_only=False)

    new = poly.fit_transform(x_train)
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(columns,p) for p in poly.powers_]]
    poly_train = pd.DataFrame(new, columns=target_feature_names)

    new = poly.fit_transform(x_test)
    target_feature_names = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(columns,p) for p in poly.powers_]]
    poly_test = pd.DataFrame(new, columns=target_feature_names)
    
    poly_train.drop('', inplace=True, axis=1)
    poly_test.drop('', inplace=True, axis=1)

    return poly_train, poly_test

if len(sys.argv) == 4:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
    PROCESSED_DIRECTORY = sys.argv[3]
else:
    IMAGE_FOLDER = '/datadrive/data/full_data/stage1'
    INPUT_LABELS = '/datadrive/data/stage1_labels.csv'
    PROCESSED_DIRECTORY = '/datadrive/output/processed_images_1/'
    #'/datadrive/project_code/cs6250_group_project/processed_images_tutorial/'
    
PROCESSED_IMAGE_BASED_NAME = "processed_patient_scan_{}.npy"
    
def load_npy(patient_id):
    return np.load(PROCESSED_DIRECTORY+PROCESSED_IMAGE_BASED_NAME.format(patient_id))

def get_patients(image_directory, labels_path):
    """
    Load list of train and test patients. For train patients, return outcome label along with patient id
    :param image_directory: directory containing images
    :param labels_path: path to training dataset labels
    :return: 2 pandas dataframes, each contains id and cancer indicator
                1st contains train patients [patient id and cancer indicator (0/1)]
                2nd contains test patients [patient id and cancer indicator is null]
    """
    patients = os.listdir(image_directory)
    train_labels = pd.read_csv(labels_path)
    patients_df = pd.DataFrame({'id': patients})
    patients_df = pd.merge(patients_df, train_labels, how='left', on='id')
    patients_df = patients_df.reindex(np.random.permutation(patients_df.index))
    train_patients = patients_df[pd.notnull(patients_df['cancer'])]
    test_patients = patients_df[pd.isnull(patients_df['cancer'])]
    return patients_df, train_patients, test_patients


data_path = '/datadrive/output/supplemental_data/complete_supplemental_dataset.csv'
df = pd.read_csv(data_path)
df.drop(['Unnamed: 0'], inplace=True, axis=1)

labels_df = pd.read_csv('/datadrive/data/stage1_labels.csv')
df = pd.merge(df, labels_df, how='left', left_on='id', right_on='id')
df = df.dropna()

predictors = ['blood', 'bone', 'emphysema', 'fat', 'muscle', 'soft tissue', 'water', 'lung_volume']

X = df.copy()[predictors]
y = df['cancer']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2017)

x_train, x_test = scaledPolyFeatures(x_train, x_test, predictors)

best_models = hyperTune(x_train, y_train)

weights = [1,1,1,1,1]
y_pred_voted = voted(x_train, y_train, x_test, weights, best_models)
print(roc_auc_score(y_test, y_pred_voted))

y_pred_voted = voted_prob(x_train, y_train, x_test, weights, best_models)
print('Log-loss: {}'.format(log_loss(y_test, y_pred_voted)))

y_pred_voted = voted_pred(x_train, y_train, x_test, weights, best_models)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred_voted)))

for m in best_models:
    print(m)
    m.fit(x_train, y_train)
    y_pred = m.predict_proba(x_test)
    print('Logloss : {}'.format(metrics.log_loss(y_test, y_pred)))
    y_pred = m.predict(x_test)
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred_voted)))
