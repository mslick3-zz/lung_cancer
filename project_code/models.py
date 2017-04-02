from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,\
AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.qda import QDA

import re
import pandas as pd
import numpy as np
import itertools

def hyperTune(x_train, y_train):
    base_models = [QDA(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        LogisticRegression(),
        AdaBoostClassifier(),
        MultinomialNB()                  
    ]

    params = [
        {},
        {'n_estimators':[150,300], 'learning_rate':[.01,.1], 'max_leaf_nodes':[None]},
        {'n_estimators':[200,300], 'max_features':['auto', 'log2']},
        {'criterion':['gini', 'entropy'], 'max_depth':[10, 20]},
        {'C':[.01, .1, 10]},
        {'n_estimators':[100,200,300]},
        {'alpha':[.0001, .001, .01, .1, 1]},
        {}
    ]

    print "Starting optimization ..."
    
    best_models = []
    
    for i, model in enumerate(base_models):
        CV_rfc = GridSearchCV(estimator=model, param_grid=params[i], cv=5, scoring='neg_log_loss')
        CV_rfc.fit(x_train, y_train)
        best_models.append(CV_rfc.best_estimator_)
        
    print "Done optimizing model parameters!"
    
    return best_models
    
    
def voted(x_train, y_train, x_test, est_weights, models):
    voting_models = [(re.sub(r'[^A-Z]', '', str(m))[0:2].lower(), m) for m in models]
    vc = VotingClassifier(estimators=voting_models, voting='soft', weights=est_weights)

    vc.fit(x_train, y_train)
    Y_pred_voted = vc.predict_proba(x_test)[:, 1]
    
    return Y_pred_voted
    
def stacked(x_train, y_train, x_test, nfolds, models, stacker):
    X = x_train.values
    y = y_train.values
    T = x_test.values
    
    folds = list(KFold(len(y)), n_folds=nfolds, shuffle=True, random_state=10)
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))

    for i, clf in enumerate(models):
        S_test_i = np.zeros((T.shape[0], len(folds)))
        for j, (train_idx, test_idx) in enumerate(folds):
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = clf.predict(T)[:]
            
        S_test[:, i] = S_test_i.mean(1)
        
    stacker.fit(S_train, y_train)
    
    Y_pred = stacker.predict_proba(S_test)[:, 1]
    
    return Y_pred


#%% Sample run with train/test data

best_models = hyperTune(x_train, y_train)

weights = [1, 1, 1, 1, 1, 1, 1]
y_pred_voted = vote(x_train, y_train, x_test, weights, best_models)

print 'Voting AUC : {}'.format(roc_auc_score(y_test, y_pred_voted))


nfolds = 10
stacker = best_models[2]
y_pred_stacked = stacked(x_train, y_train, x_test, nfolds, best_models, stacker)

print 'Stacking AUC : {}'.format(roc_auc_score(y_test, y_pred_stacked))



































