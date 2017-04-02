"""
This script runs a quick analysis on features extracted during the preprocessing phase
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from matplotlib import pyplot as plt
from Inputs import *
from sklearn.metrics import confusion_matrix, log_loss
import sys

if len(sys.argv) == 3:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
else:
    IMAGE_FOLDER = '../../../full_set/stage1/'
    INPUT_LABELS = '../../../input/stage1_labels.csv'

def supplemental_data_analysis():
    base_dir = '../supplemental_data/'
    files = os.listdir(base_dir)
    data = pd.read_csv(os.path.join(base_dir, files[0]), index_col=False)
    for file in files[1:]:
        data = data.append( pd.read_csv(os.path.join(base_dir, file), index_col=False) )

    data.drop(data.columns[0], axis=1, inplace=True)
    all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS)
    data[['id']] = data[['id']].astype(str)
    train_patients[['id']] = train_patients[['id']].astype(str)
    data = pd.merge(data, train_patients, on='id')
    data.set_index('id',inplace=True)

    train_x, valid_x, train_y, valid_y = train_test_split(data.drop('cancer', axis=1),
                                                        data.cancer,
                                                        random_state=12345,
                                                        train_size=0.7,
                                                        stratify=data.cancer)

    clf = xgb.XGBRegressor(max_depth=5,
                           n_estimators=10,
                           min_child_weight=1,
                           learning_rate=0.05,
                           nthread=-1,
                           subsample=0.80,
                           colsample_bytree=1.0,
                           seed=4242)

    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='logloss', early_stopping_rounds=50)

    all_preds = clf.predict(data.drop('cancer', axis=1))
    all_conf_mat = confusion_matrix(np.round(all_preds, 0).astype(np.int), data.cancer)
    print(all_conf_mat) #all data predictions
    print(log_loss(data.cancer.astype(np.int), all_preds))

    valid_preds = clf.predict(valid_x)
    valid_conf_mat = confusion_matrix(np.round(valid_preds, 0).astype(np.int), valid_y.astype(np.int))
    print(valid_conf_mat) #validation data predictions
    print(log_loss(valid_y.astype(np.int), valid_preds))

    print(log_loss(valid_y.astype(np.int), np.repeat(.0, len(valid_y))))
    print(log_loss(valid_y.astype(np.int), np.repeat(.5, len(valid_y))))

    return

if __name__ == '__main__':
    supplemental_data_analysis()