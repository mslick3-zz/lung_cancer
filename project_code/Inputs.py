"""
This script reads in files - either complete scans or the labeled training file
"""
import numpy as np # linear algebra
import dicom
import os
import SimpleITK as sitk
import pandas as pd

def load_scan(path):
    """
    Load all slices and metadata from dicom images in path to patient
    :param path: path to a patient's scans
    :return: the metadata for the image slices
    """
    slices = [dicom.read_file(os.path.join(path, filename)) for filename in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

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