"""
This script get patients 
"""
import numpy as np 
import os
import pandas as pd

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
