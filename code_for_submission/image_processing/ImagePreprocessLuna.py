"""
This script runs the preprocessing on all images - train and test. set the folder for images and input label file path
"""
import os
from Inputs import *
from ImageProcessUtils import *
import multiprocessing as mp
import warnings
from LUNAProcesses import *

warnings.filterwarnings("ignore")

import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('input_path', action='store', help='input path')

parser.add_argument('output_path', action='store', help='output path', default='../../../output_luna/')

parser.add_argument('shape', action='store', help='output shape', default='512,512')

parser.add_argument('method', action='store', help='method', default='5')

params = parser.parse_args()

INPUT_FOLDER = params.input_path
IMAGE_SIZE = tuple(np.array(params.shape.split(',')).astype(np.int))
METHOD = [5]
OUTPUT_DIR = params.output_path

def process_scan(patient):
    """
    Pre-processing steps taken for each patient
    :param patient_id: patient id (subfolder name)
    :return: supplemental data
    """
    if patient['diameter_mm']:
        cls = 1
    else:
        cls=0
    patient_id = os.path.basename(os.path.normpath(patient['directory']))
    patient_scan_data = load_scan(patient['directory'])  # load all dicom files for patient
    hu_pixels = get_pixels_hu(patient_scan_data) #translate each slice to HU
    n_scans = hu_pixels.shape[0] #number of image slices in scan
    #collect supplemental data for middle image in the scan
    supplemental_data = extra_features(hu_pixels[int(n_scans/2),:,:], patient_scan_data, patient_scan_data[int(n_scans/2)].PixelSpacing) #supplemental features
    hu_pixels = resample(hu_pixels, patient_scan_data) #resample the data to account for varying pixel sizes

    spacing = np.array([patient_scan_data[0].PixelSpacing + [patient_scan_data[0].SliceThickness]]).flatten()
    origin = np.array(patient_scan_data[0].ImagePositionPatient[:2] + [min(patient_scan_data,key= lambda x: x.ImagePositionPatient[2]).ImagePositionPatient[2]])
    center = np.array([patient['coordX'], patient['coordY'], patient['coordZ']])
    v_center = np.rint((center-origin)/spacing).astype(np.int).flatten()

    segmented, lung_volume = lung_segmentation(hu_pixels[(v_center[2]-2):(v_center[2]+3),:,:], IMAGE_SIZE, METHOD, normalize_image=True) #get segmented images and lung volume
    supplemental_data['lung_volume'] = [lung_volume] #append lung volume to supllemental dataset
    supplemental_data['id'] = [patient_id]

    num_z, height, width = hu_pixels.shape
    mask = make_mask(center, patient['diameter_mm'], width, height, spacing, origin, 5, v_center, num_z, segmented.get(5).shape)

    np.save(os.path.join(OUTPUT_DIR, "class_{}_processed_images_".format(cls) + str(5),'processed_patient_scan_{}.npy'.format(patient_id)), segmented[5]) #save processed image
    np.save(os.path.join(OUTPUT_DIR, "class_{}_masked_images_".format(cls) + str(5),'processed_patient_mask_{}.npy'.format(patient_id)), mask) #save processed image


    unsegmented = resize(hu_pixels, segmented.get(METHOD[0]).shape)
    np.save(os.path.join(OUTPUT_DIR, "processed_images_unsegmented",'processed_patient_scan_{}.npy'.format(patient_id)),unsegmented)  # save processed image
    supplemental_data.to_csv(os.path.join(OUTPUT_DIR,"supplemental_data","processed_patient_supplemental_{}.csv".format(patient_id))) #save supplemental data
    return supplemental_data

def run():
    """
    Driver method that runs the preprocessing steps
    :return: None
    """
    #get train and test patient lists

    #if directories don't exist, make them
    for method in METHOD:
        if not os.path.isdir(os.path.join(OUTPUT_DIR,"class_0_processed_images_"+str(method))):
            os.makedirs(os.path.join(OUTPUT_DIR,"class_0_processed_images_"+str(method)))
        if not os.path.isdir(os.path.join(OUTPUT_DIR, "class_1_processed_images_" + str(method))):
            os.makedirs(os.path.join(OUTPUT_DIR, "class_1_processed_images_" + str(method)))
        if not os.path.isdir(os.path.join(OUTPUT_DIR, "class_0_masked_images_" + str(method))):
            os.makedirs(os.path.join(OUTPUT_DIR, "class_0_masked_images_" + str(method)))
        if not os.path.isdir(os.path.join(OUTPUT_DIR, "class_1_masked_images_" + str(method))):
            os.makedirs(os.path.join(OUTPUT_DIR, "class_1_masked_images_" + str(method)))
    if not os.path.isdir(os.path.join(OUTPUT_DIR, "processed_images_unsegmented")):
        os.makedirs(os.path.join(OUTPUT_DIR, "processed_images_unsegmented"))
    if not os.path.isdir(os.path.join(OUTPUT_DIR,"supplemental_data")):
        os.makedirs(os.path.join(OUTPUT_DIR,"supplemental_data"))

    all_patients = get_patients_luna(INPUT_FOLDER)

    all_patients_list = []
    for index, patient in all_patients.iterrows():
        all_patients_list.append(patient)

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)  # Create a multiprocessing Pool
    data = pool.map(process_scan, all_patients_list)  # proces data_inputs iterable with pool
    pool.close()

    supplemental_data = data[0]
    for df in data[1:]:
        supplemental_data = supplemental_data.append(df)
    supplemental_data.to_csv(os.path.join(OUTPUT_DIR,"full_supplemental_data_luna","complete_supplemental_dataset.csv"))

    return

if __name__ == '__main__':
    """
    Procesing Full Dataset - Train and Test Images
    """
    run()
