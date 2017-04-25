"""
This script reads in files - either complete scans or the labeled training file
"""
import numpy as np # linear algebra
import dicom
import os
import SimpleITK as sitk
import pandas as pd
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
import LUNAProcesses

def load_scan(path):
    """
    Load all slices and metadata from dicom images in path to patient
    :param path: path to a patient's scans
    :return: the metadata for the image slices
    """
    slices = [dicom.read_file(os.path.join(path, filename)) for filename in os.listdir(path) if filename.endswith('.dcm')]
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


def get_patients_luna(luna_path):
    """
    read annotations file for LUNA16 data
    source: https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    :param path: directory of annotations file
    :return: The locations of the nodes
    """

    image_path = os.path.join(luna_path,'DOI')
    files_path = os.path.join(luna_path,'CSVFILES')
    file_list = glob(os.path.join(image_path, "**", "**", "**"))

    def get_filename(case):
        for f in file_list:
            if case in f:
                return f
        return None

    df_node = pd.read_csv(os.path.join(files_path , "annotations.csv"))
    df_node["directory"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()

    df_dud = pd.read_csv(os.path.join(files_path, "candidates.csv"))
    df_dud.drop('class', axis=1, inplace=True)
    df_node = df_node.append(df_dud.head(df_node.shape[0]))

    return df_node