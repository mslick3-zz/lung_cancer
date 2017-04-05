"""
This script runs the preprocessing on all images - train and test. set the folder for images and input label file path
"""
import os
from ImageProcessUtilsTutorial import *
import multiprocessing as mp
import warnings
from Inputs import *
warnings.filterwarnings("ignore")

import sys

if len(sys.argv) == 3:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
else:
    IMAGE_FOLDER = '/Users/adnan/Development/git/masters/bd4h/group_project/sample_images/'
    INPUT_LABELS = '/Users/adnan/Development/git/masters/bd4h/group_project/stage1_labels.csv'

IMAGE_SIZE = (120, 120)
IMAGE_DEPTH = 30
IMG_SIZE_PX = 50
SLICE_COUNT = 20

def process_scan(patient_id):
    """
    Pre-processing steps taken for each patient
    :param patient_id: patient id (subfolder name)
    """
    patient_scan_data = process_data(patient_id, IMAGE_FOLDER, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
    np.save('../processed_images_tutorial/processed_patient_scan_{}.npy'.format(patient_id), patient_scan_data) #save processed image
    return

def run():
    """
    Driver method that runs the preprocessing steps
    :return: None
    """
    #get train and test patient lists
    all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS)

    #if directories don't exist, make them
    if not os.path.isdir("../processed_images_tutorial"):
        os.makedirs("../processed_images_tutorial")

    #run the patient preprocessing, 1 patient per CPU core available
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)  # Create a multiprocessing Pool
    data = pool.map(process_scan, all_patients['id'])  # proces data_inputs iterable with pool
    pool.close()

    return

if __name__ == '__main__':
    """
    Procesing Full Dataset - Train and Test Images
    """
    run()
