import SimpleITK as sitk
import os
import numpy as np
import csv
import scipy
from glob import glob
import pandas as pd
from scipy import ndimage
from tqdm import tqdm 
import pandas as pd
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from skimage import measure, morphology, segmentation

from joblib import Parallel, delayed
from numba import autojit
import zarr
from ConfigParser import SafeConfigParser
from PIL import Image
import cv2
from kaggle_utils import *


def load_train():
    patients = os.listdir(src)
    patients.sort()
    return patients
        
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    pos1 = slices[int(len(slices)/2)].ImagePositionPatient[2]
    pos2 = slices[(int(len(slices)/2)) + 1].ImagePositionPatient[2]
    diff = pos2 - pos1
    if diff > 0:
        slices = np.flipud(slices)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
    
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image <= threshold_min] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)
    
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    if spacing[0] == 0.0:
        spacing[0] = spacing[1]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing



def reshape_3d(image_3d):
    reshaped_img = image_3d.reshape([image_3d.shape[0], 1, img_size, img_size])
    print('Reshaped image shape:', reshaped_img.shape)
    return reshaped_img


def create_masks_for_patient_watershed(patient_id, save = False):

    print("Getting mask for patient {}".format(patient_id))
    dicom_file = src + patient_id
    patient = load_scan(dicom_file)
    lung_orig = get_pixels_hu(patient)
    lung_img, spacing = resample(lung_orig, patient, RESIZE_SPACING)

    lung_mask = lung_img.copy()
    lung_mask[lung_mask >= threshold_max] = threshold_max
    lung_img[lung_img >= threshold_max] = threshold_max

    lung_img_512, lung_mask_512 = np.zeros((lung_img.shape[0], img_size, img_size)), np.zeros((lung_mask.shape[0], img_size, img_size))
    lung_mask_512[lung_mask_512 == 0] = threshold_min
    lung_img_512[lung_img_512 == 0] = threshold_min

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (img_size - original_shape[1])
        upper_offset = int(np.round(offset/2))
        lower_offset = int(offset - upper_offset)
        lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]

    lung_mask_pres = reshape_3d(lung_mask_512)
    lung_raw_pres = reshape_3d(lung_img_512)
    lung_mask_pres[lung_mask_pres <= threshold_min] = threshold_min
    
    lung_mask_preproc = my_PreProc(lung_mask_pres)
    lung_mask_preproc = lung_mask_preproc.astype(np.float32)
    print(lung_mask_preproc.shape)
    
    if save:
        save_zarr(patient_id, lung_mask_preproc)
        
    return


def save_zarr(id_patient, lung_mask):
    lung_mask_group.array(id_patient, lung_mask, 
            chunks=(10, 1, img_size, img_size), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    return

config = SafeConfigParser()
config.read('../../file_paths.config')

src = config.get('main', 'kaggle_dataset_images')
processed_img_dst = config.get('main', 'kaggle_processed_image_output')
print(processed_img_dst) 
#create directory if absent 
if not os.path.exists(processed_img_dst + "/lung_mask"):
    os.makedirs(processed_img_dst + "/lung_mask")

zarr_store = zarr.DirectoryStore(processed_img_dst)
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
lung_mask_group = zarr_group.require_group('lung_mask')


patients = load_train()

threshold_min = -2000
threshold_max = 400
img_size = 168
RESIZE_SPACING = [3, 3, 3]

def run():
    for i in patients:
        create_masks_for_patient_watershed(i, True)
    return

run()
