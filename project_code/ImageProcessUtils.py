"""
This script has the image preprocessing utility functions
"""
from scipy import ndimage
from skimage import measure, feature, morphology, segmentation
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import numpy as np
import cv2
import pandas as pd

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def normalize(image):
    """
    Min-max image scaling
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param image: a numpy array containing image pixel values
    :return: a numpy array containing the image pixel values normalized in the range [MIN_BOUND, MAX_BOUND]
    """
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>(1-PIXEL_MEAN)] = 1.
    image[image<(0-PIXEL_MEAN)] = 0.
    return image

def resample(image, scan):
    """
    Resampling to account for different image resolutions
    :param image: a 3d numpy array containing the image pixel values
    :param scan: metadata about the image
    :return: 3d numpy array of the resampled scan
    """
    # Determine current pixel spacing
    resize_factor = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    new_shape = np.round(image.shape * resize_factor)
    real_resize_factor = new_shape / image.shape

    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def audit_segmentation(im):
    """
    Performs check on image segmentation as a quick check to make sure there wasn't a serious error
    :param im: a numpy array containing the image pixel values for a binary image
    :return: tuple containing the %pixels that are 0/1 and assessment of segmenting algorithm
    """
    good_segmentation = True
    vals, counts = np.unique(im.flatten(), return_counts=True)
    counts = counts/np.sum(counts)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_as_ubyte(im), 8, cv2.CV_32S)

    if num_labels not in [2, 3]:
        good_segmentation = False
    elif np.amax(counts) > 0.9:
        good_segmentation = False
    elif counts[0] > 0.9:
        good_segmentation = False

    return counts, good_segmentation

def get_dicom_info(slices):
    """
    Get some attributes from the DICOM files
    :param slices: dicom files for patient
    :return: dictionary of attributes - age, gender, etc.
    """
    attributes = {'age': slices[0].get('PatientAge'),
                  'gender': slices[0].get('PatientSex')
                  }
    return attributes

def get_pixels_hu(slices):
    """
    Convert pixel values to HU scale
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param slices: list of dicom images and metadata
    :return: a 3d numpy array of pixel values for all slices for the patient
    """
    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    outside_image = np.amin(image)
    image[image == outside_image] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def watershed_lung_segmentation_image(image):
    """
    A method using watershed algorithm to perform segmentation. seems to be pretty accurate, but is really slow
    :source: https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed
    :param image: 2d numpy array of an image slice
    :return:
    """
    def generate_markers(image):
        # Creation of the internal Marker
        marker_internal = image < -400
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0
        marker_internal = marker_internal_labels > 0
        # Creation of the external Marker
        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a
        # Creation of the Watershed Marker matrix
        marker_watershed = np.zeros(image.shape, dtype=np.int)
        marker_watershed += marker_internal * image.shape[0]
        marker_watershed += marker_external * int(np.ceil(image.shape[1]/2.))

        return marker_internal, marker_external, marker_watershed

    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= image.shape[0] / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones(image.shape))

    return segmented, lungfilter

def fast_lung_segment(image, fill_lung_structures=True):
    """
    My attempts to improve segmented_lung_mask. Tweaked code above using some other examples online and info from CV/CP
    Perform segmentation of lungs in image much better and faster than method above
    :param image: numpy array of pixel values
    :param fill_lung_structures: boolean of whether to perform connected component algorithm on image
    :return: numpy array representing a binary image of thresholded pixel values
    """
    binary_image = cv2.threshold(image, -300, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    im2, contours, _ = cv2.findContours(binary_image,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(binary_image.shape, np.uint8)
    cv2.fillPoly(mask, [largest_contour], 255)

    binary_image = ~binary_image
    binary_image[(mask == 0)] = 0

    # apply closing to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)
    image[binary_image==0] = 0
    return image, binary_image

def lung_segmentation(scan, resize_image, method=1, depth=30, normalize_image=True):
    """
    Runs segment_lung_mask3 for all images in a scan
    :param scan: 3d numpy array containing image scan pixel values
    :param depth: returns the size of the stack of image slices returned - 'median' for middle slice, int for # of most filled slices, None for average of all slices
    :param resize_image: tuple of desired image shape.  None for the original size
    :param normalize_image: boolean of whether the image should be normalized. min-max scaling and 0-centered
    :return: 2d or 3d numpy array with lungs segmented, also returns lung volume for patient
    """
    if resize_image is not None:
        segmented = np.zeros((scan.shape[0],)+resize_image)
    else:
        segmented = np.zeros(scan.shape)
    lung_volumes = np.zeros(scan.shape[0])
    for i in range(scan.shape[0]):
        segment, filter = fast_lung_segment(scan[i, :, :])
        counts, good_segmentation = audit_segmentation(filter)
        if not good_segmentation:
            segment, filter = watershed_lung_segmentation_image(scan[i, :, :])
            counts, good_segmentation = audit_segmentation(filter)
        if resize_image is not None:
            segment = cv2.resize(segment, resize_image, interpolation=cv2.INTER_CUBIC)
        if normalize_image:
            segment = normalize(segment)
        segmented[i,:,:] = segment
        lung_volumes[i] = counts[0]

    lung_volume = np.sum(1. - lung_volumes)

    segmented_out = {1:None, 2:None, 3:None, 4:None, 5:None, 6:None}

    if 1 in method:
        if depth > 0: #return image stack of most filled in lungs of specified depth
            lung_volume_ma = pd.rolling_mean(lung_volumes, depth)
            idx = np.argmin(lung_volume_ma[depth:]) + depth
            lower = idx
            upper = min(idx+depth, segmented.shape[0]) +4

            if min(upper, segmented.shape[0]) - idx != depth:
                upper = int(segmented.shape[0]/2)+15
                lower = int(segmented.shape[0]/2)-15

            segmented_out[1] = segmented[lower:upper,:,:]
    if 2 in method: #return average of all slices
        segmented_out[2] = np.mean(segmented, axis=0)
    if 3 in method: #return only middle slice
        segmented_out[3] = segmented[int(segmented.shape[0]/2.),:,:]
    if 4 in method:
        idx = np.unique(np.linspace(0,segmented.shape[0],num=depth+1).astype(np.int))
        segmented_accumulator = np.zeros((depth,)+resize_image)
        for i in range(len(idx)-1):
            lower = idx[i]
            upper = idx[i+1]
            segmented_accumulator[i, :, :] = np.mean(segmented[lower:upper,:,:], axis=0)
        segmented_out[4] = segmented_accumulator
    if 5 in method:
        segmented_out[5] = segmented
    if 6 in method:
        segmented_out[6] = []
        seq = np.arange(0, segmented.shape[0],3)
        for i in range(len(seq)-1):
            segmented_out[6].append(segmented[seq[i]:seq[i+1],:,:])

    return segmented_out, lung_volume

def extra_features(slice, scan, pixel_spacing):
    """
    Method calculated the amount of blood, water, and fat in an image
    Values from https://en.wikipedia.org/wiki/Hounsfield_scale
    :source: Idea expanded from https://www.kaggle.com/kmader/data-science-bowl-2017/simple-single-image-features-for-cancer
    :param in_dcm: 2d numpy array. image slice
    :return: dataframe with volume in image slice containing each feature
    """
    feature_list = {
    'blood': (30, 45),
    'bone': (700, 3000),
    'emphysema': (900, 950),
    'fat': (-100, -50),
    'muscle': (10, 40),
    'soft tissue': (100, 300),
    'water': (-10, 10)
    }
    pix_area = np.prod(pixel_spacing)

    features = {}

    for key, value in feature_list.items():
        features[key] = [pix_area * np.sum((slice >= value[0]) & (slice <= value[1]))]

    other_features = get_dicom_info(scan)
    features.update(other_features)

    features = pd.DataFrame(features)

    return features


def make_mask(center,diam,width,height,spacing,origin,depth, v_center, num_z, reshape=None):
    """
    source: https://github.com/booz-allen-hamilton/DSB3Tutorial/blob/master/tutorial_code/LUNA_mask_extraction.py
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    """
    masks = np.zeros([depth, height, width])
    def get_2d_mask(center, diam, z, width, height, spacing, origin):
        mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
        # convert to nodule space from world coordinates

        # Defining the voxel range in which the nodule falls
        v_center = (center - origin) / spacing
        v_diam = int(diam / spacing[0] + 5)
        v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
        v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
        v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
        v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

        v_xrange = range(v_xmin, v_xmax + 1)
        v_yrange = range(v_ymin, v_ymax + 1)

        # Fill in 1 within sphere around nodule
        for v_x in v_xrange:
            for v_y in v_yrange:
                p_x = spacing[0] * v_x + origin[0]
                p_y = spacing[1] * v_y + origin[1]
                if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                    mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
        return mask

    for i, i_z in enumerate(np.arange(int(v_center[2]) - 1, int(v_center[2]) + 2).clip(0, num_z - 1)):  # clip prevents going out of bounds in Z
        mask = get_2d_mask(center, diam, i_z * spacing[2] + origin[2], width, height, spacing, origin)
        masks[i,:,:] = mask

    if reshape is not None:
        zoom = np.array(reshape)/np.array(masks.shape)
        masks = ndimage.interpolation.zoom(masks, zoom, mode='nearest')
        masks[masks>0.1] = 1.0
        masks[masks!=1.0] = 0.0

    return masks

def resize(image, output_size):
    """
    resize an image
    :param image: input array
    :param output_size: output size (tuple). same dimension as input size
    :return: resized image array
    """
    zoom = np.array(output_size) / np.array(image.shape)
    return ndimage.interpolation.zoom(image, zoom, mode='nearest')
