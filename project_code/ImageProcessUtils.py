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
    :return: a numpy array containing the image pixel values normalized in the range [0, 1]
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

def lung_segmentation(scan, depth, resize_image, normalize_image=True):
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

    lung_volume = np.amax(1. - lung_volumes)

    if type(depth) == int:
        if depth > 0: #return image stack of most filled in lungs of specified depth
            lung_volume_ma = pd.rolling_mean(lung_volumes, depth)
            idx = np.argmin(lung_volume_ma[depth:]) + depth
            segmented = segmented[idx:(idx+depth),:,:]
    else:
        if depth is None: #return average of all slices
            segmented = np.mean(segmented, axis=0)
        elif depth=='median': #return only middle slice
            segmented = segmented[int(segmented.shape[0]/2.),:,:]

    return segmented, lung_volume

def extra_features(slice, pixel_spacing):
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

    features = pd.DataFrame(features)

    return features


