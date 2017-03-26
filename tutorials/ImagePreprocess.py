import os
from ImageReader import *
from ImageProcessUtils import *
from ImagePlots import *

INPUT_FOLDER = '../input/sample_images/'


def tutorial1():
    '''
    sample processing shown in turorial
    '''
    #first tutorial https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()

    first_patient = load_scan(INPUT_FOLDER + patients[0])
    first_patient_pixels = get_pixels_hu(first_patient)
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

    segmented_lungs = segment_lung_mask2(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask2(pix_resampled, True)

    # plot_histogram_scan(first_patient_pixels)
    # plot_2d_slice(first_patient_pixels[80])


def tutorial2():
    '''
    sample processing shown in turorial
    '''
    # second tutorial https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    ct_scan = load_scan('../input/sample_images/00cba091fa4ad62cc3200a657aeb957e/')
    images = np.dstack([img.pixel_array for img in ct_scan])
    #plot_grid_scan(images)
    get_segmented_lungs(images[72], True)
    segmented_ct_scan = segment_lung_from_ct_scan(images)
    #plot_grid_scan(segmented_ct_scan)
    segmented_ct_scan[segmented_ct_scan < 604] = 0
    #plot_grid_scan(segmented_ct_scan)

    #plot_3d_scan(first_patient_pixels,-300,'test.png')
    #plot_3d_scan(segmented_lungs_fill,'test3.png')


if __name__ == '__main__':
    """
    Procesing Full Dataset
    """
    # get list of patients from file directory
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()

    for patient_id in patients:
        patient_scan_data = load_scan(os.path.join(INPUT_FOLDER, patient_id)) #load all dicom files for patient
        # get image data and stack the 2d slices to form 3d array. convert from RGB to HU also
        patient_pixels = get_pixels_hu(patient_scan_data)
        pix_resampled, spacing = resample(patient_pixels, patient_scan_data, [1,1,1]) #resample to adjust for reesolution variations

        segmented_lungs_fill = segment_lung_mask2(pix_resampled, True) #perfrom segmentation
        segmented_lungs_fill = segment_lung_mask3(patient_pixels[55,:,:], True)[0]  # perfrom segmentation

        plot_2d_slice(segmented_lungs_fill)
