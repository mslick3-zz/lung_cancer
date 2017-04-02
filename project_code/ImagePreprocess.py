import os
from Inputs import *
from ImageProcessUtils import *
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

IMAGE_FOLDER = '../../../full_set/stage1/'
INPUT_LABELS = '../../../input/stage1_labels.csv'
IMAGE_SIZE = (120, 120)
IMAGE_DEPTH = 30

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
    supplemental_data = extra_features(hu_pixels[int(n_scans/2),:,:], patient_scan_data[int(n_scans/2)].PixelSpacing) #supplemental features
    hu_pixels = resample(hu_pixels, patient_scan_data) #resample the data to account for varying pixel sizes
    segmented, lung_volume = lung_segmentation(hu_pixels, IMAGE_DEPTH, IMAGE_SIZE, normalize_image=True) #get segmented images and lung volume
    supplemental_data['lung_volume'] = [lung_volume] #append lung volume to supllemental dataset
    np.save('../processed_images/processed_patient_scan_{}.npy'.format(patient_id), segmented) #save processed image
    supplemental_data.to_csv('../supplemental_data/processed_patient_supplemental_{}.npy'.format(patient_id)) #save supplemental data
    return supplemental_data

def run():
    """
    Driver method that runs the preprocessing steps
    :return: None
    """
    #get train and test patient lists
    all_patients, train_patients, test_patients = get_patients(IMAGE_FOLDER, INPUT_LABELS)

    #if directories don't exist, make them
    if not os.path.isdir("../processed_images"):
        os.makedirs("../processed_images")
    if not os.path.isdir("../supplemental_data"):
        os.makedirs("../supplemental_data")

    #run the patient preprocessing, 1 patient per CPU core available
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)  # Create a multiprocessing Pool
    data = pool.map(process_scan, all_patients['id'][:3])  # proces data_inputs iterable with pool
    pool.close()

    #append all supplemental data to a final, complete dataset
    supplemental_data = data[0]
    for df in data[1:]:
        supplemental_data = supplemental_data.append(df)
    supplemental_data.to_csv('../supplemental_data/complete_supplemental_dataset.csv')

    return

if __name__ == '__main__':
    """
    Procesing Full Dataset - Train and Test Images
    """
    run()
