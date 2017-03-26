from scipy import ndimage
from skimage.morphology import ball, disk, binary_erosion, binary_closing
from skimage.measure import label,regionprops
from skimage.filters import roberts
from skimage import measure, feature, morphology, segmentation
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def zero_center(image):
    """
    Center the image at 0
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param image: a numpy array containing the image pixel values
    :return: a numpy array containing the images shited to contain a mean value of 0
    """
    image = image - PIXEL_MEAN
    return image

def resample(image, scan, new_spacing=[1, 1, 1]):
    """
    Resampling to account for different image resolutions
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param image: a 3d numpy array containing the image pixel values
    :param scan: metadata about the image
    :param new_spacing: new image pixel spacing. default set to [1, 1, 1]
    :return: tuple containing the resampled image and the new spacing
    """
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def largest_label_volume(im, bg=-1):
    """
    Performs connected component operation to segment the lungs from the background
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param im: a numpy array containing the image pixel values
    :param bg: background threshold
    :return: the most frequent value occuring in the image that !=bg
    """
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

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
    outside_image = image.min()
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


def segment_lung_mask(image, fill_lung_structures=True):
    """
    Perform segmentation of lungs in image
    :source: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook
    :param image: 3d numpy array of pixel values
    :param fill_lung_structures: boolean of whether to perform connected component algorithm on image
    :return: 3d numpy array representing a binary image of thresholded pixel values
    """
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

def segment_lung_mask2(image, fill_lung_structures=True):
    """
    My attempts to improve segmented_lung_mask. Tweaked code above using some other examples online and info from CV/CP
    Perform segmentation of lungs in image much better and faster than method above
    :param image: numpy array of pixel values
    :param fill_lung_structures: boolean of whether to perform connected component algorithm on image
    :return: numpy array representing a binary image of thresholded pixel values
    """
    binary_image_scan = np.zeros(image.shape)
    binary_images = cv2.threshold(image, -300, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    for i in range(image.shape[0]):
        binary_image = binary_images[i,:,:]
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

        binary_image_scan[i,:,:] = binary_image

    return binary_image_scan

def segment_lung_mask3(image, fill_lung_structures=True):
    """
    A method using watershed algorithm to perform segmentation. seems to be a lot more accurate, but is really slow
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
        marker_watershed = np.zeros((512, 512), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128

        return marker_internal, marker_external, marker_watershed

    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

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
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))

    return segmented, lungfilter

def segment_lung_mask3_all(image, fill_lung_structures=True):
    """
    Runs segment_lung_mask3 for all images in a scan
    :param image: 3d numpy array containing image scan pixel values
    :param fill_lung_structures: ignored
    :return: 3d numpy array with lungs segmented
    """
    segmented = np.zeros(image.shape)
    lungfilter = np.zeros(image.shape)
    for i in range(image.shape[0]):
        segment, filter = segment_lung_mask3(image[i, :, :], True)
        segmented[i,:,:] = segment
        lungfilter[i,:,:] = filter
    return segmented, lungfilter

def get_segmented_lungs(im, plot=False):
    """
    Performs segmentation from a 2d scan slice
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param im: a 2d numpy array of pixel values representing a scan slice
    :param plot: boolean of whether a plot should be generated in window
    :return: a 2d numpy array of binary pixel values representing a segmented image
    """
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    return im

def segment_lung_from_ct_scan(ct_scan):
    """
    Performs segmentation on each image slice in a scan
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param ct_scan: list of slices for a patient
    :return: 3d numpy array of segmented slices
    """
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def connected_components(segmented_scan):
    """
    A basic method for performing connected component analysis on an image
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param image: a 2d numpy array containing a binary image's pixel values
    :return: 2d numpy array with connected components labeled
    """
    selem = ball(2)
    binary = binary_closing(segmented_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (
            min((max_x - min_x), (max_y - min_y), (max_z - min_z)))






