"""
This script runs the preprocessing on all images - train and test. set the folder for images and input label file path
"""
import os
from Inputs import *
from ImageProcessUtils import *
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import sys

print(sys.argv)

if len(sys.argv) >= 5:
    IMAGE_FOLDER = sys.argv[1]
    INPUT_LABELS = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]
    METHOD = np.array(sys.argv[4].split(',')).astype(np.int8)
else:
    print('enter valid set of parameters')
    exit(1)

IMAGE_SIZE = (512, 512)

def process_scan(patient_id):
    """
    Pre-processing steps taken for each patient
    :param patient_id: patient id (subfolder name)
    :return: supplemental data
    """
    patient_scan_data = load_scan(os.path.join(IMAGE_FOLDER, patient_id))  # load all dicom files for patient
    hu_pixels = get_pixels_hu(patient_scan_data) #translate each slice to HU
    n_scans = hu_pixels.shape[0] #number of image slices in scan
    #collect supplemental data for middle image in the scan
    supplemental_data = extra_features(hu_pixels[int(n_scans/2),:,:], patient_scan_data, patient_scan_data[int(n_scans/2)].PixelSpacing) #supplemental features
    hu_pixels = resample(hu_pixels, patient_scan_data) #resample the data to account for varying pixel sizes
    segmented, lung_volume = lung_segmentation(hu_pixels, IMAGE_SIZE, METHOD, normalize_image=True) #get segmented images and lung volume
    supplemental_data['lung_volume'] = [lung_volume] #append lung volume to supllemental dataset
    supplemental_data['id'] = [patient_id]
    for method in METHOD:
        if method == 6:
            for i in range(len(segmented[6])):
                np.save(os.path.join(OUTPUT_DIR, "full_processed_images_" + str(method),'processed_patient_scan_{}_slice_{}.npy'.format(patient_id, i)), segmented[method][i])
        else:
            np.save(os.path.join(OUTPUT_DIR, "full_processed_images_" + str(method),'processed_patient_scan_{}.npy'.format(patient_id)), segmented[method])  # save processed image

    supplemental_data.to_csv(os.path.join(OUTPUT_DIR,"full_supplemental_data","processed_patient_supplemental_{}.csv".format(patient_id))) #save supplemental data
    return supplemental_data

def run():
    """
    Driver method that runs the preprocessing steps
    :return: None
    """
    #get train and test patient lists
    all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS)

    #if directories don't exist, make them
    for method in METHOD:
        if not os.path.isdir(os.path.join(OUTPUT_DIR,"full_processed_images_"+str(method))):
            os.makedirs(os.path.join(OUTPUT_DIR,"full_processed_images_"+str(method)))
    if not os.path.isdir(os.path.join(OUTPUT_DIR,"full_supplemental_data")):
        os.makedirs(os.path.join(OUTPUT_DIR,"full_supplemental_data"))

    #run the patient preprocessing, 1 patient per CPU core available
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)  # Create a multiprocessing Pool
    data = pool.map(process_scan, all_patients['id'])  # proces data_inputs iterable with pool
    pool.close()

    #append all supplemental data to a final, complete dataset
    supplemental_data = data[0]
    for df in data[1:]:
        supplemental_data = supplemental_data.append(df)
    supplemental_data.to_csv(os.path.join(OUTPUT_DIR,"full_supplemental_data","complete_supplemental_dataset.csv"))

    return

if __name__ == '__main__':
    """
    Procesing Full Dataset - Train and Test Images
    """
    run()
