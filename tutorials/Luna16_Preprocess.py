import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage
from ImageReader import load_itk
from ImagePreprocess import segment_lung_from_ct_scan
import pickle
RESIZE_SPACING = [1, 1, 1]

def world_2_voxel(world_coordinates, origin, spacing):
    """
    This function is used to convert the world coordinates to voxel coordinates using the origin and spacing of the ct_scan
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param world_coordinates:
    :param origin:
    :param spacing:
    :return:
    """
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    """
    This function is used to convert the voxel coordinates to world coordinates using the origin and spacing of the ct_scan.
    :source: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    :param voxel_coordinates:
    :param origin:
    :param spacing:
    :return:
    """
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def seq(start, stop, step=1):
    """

    :param start:
    :param stop:
    :param step:
    :return:
    """
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])

def draw_circles(image, cands, origin, spacing):
    """
    This function is used to create spherical regions in binary masks at the given locations and radius.
    :param image:
    :param cands:
    :param origin:
    :param spacing:
    :return:
    """
    # make empty matrix, which will be filled with the mask

    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for ca in cands.values:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, spacing)
                    if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[np.round(coords[0]), np.round(coords[1]), np.round(coords[2])] = int(1)

    return image_mask

def create_nodule_mask(imagePath, maskPath, cands):
    """
    This function takes the path to a '.mhd' file as input and
    is used to create the nodule masks and segmented lungs after
    rescaling to 1mm size in all directions. It saved them in the .npz
    format. It also takes the list of nodule locations in that CT Scan as
    input.
    :param imagePath:
    :param maskPath:
    :param cands:
    :return:
    """
    # if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)
    imageName = os.path.basename(imagePath).split('.')[0]

    # calculate resize factor
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    # resize image
    lung_img = ndimage.interpolation.zoom(img, real_resize)

    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    # create nodule mask
    nodule_mask = draw_circles(lung_img, cands, origin, new_spacing)

    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
        (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = np.round(offset / 2)
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

        lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
        lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
        nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]

        # save images.
    np.save(imageName + '_lung_img.npz', lung_img_512)
    np.save(imageName + '_lung_mask.npz', lung_mask_512)
    np.save(imageName + '_nodule_mask.npz', nodule_mask_512)

def get_patch_from_list(lung_img, coords, window_size=10):
    """
    This function takes the numpy array of CT_Scan and a list of coords from
    which voxels are to be cut. The list of voxel arrays are returned. We keep the
    voxels cubic because per pixel distance is same in all directions.
    :param lung_img:
    :param coords:
    :param window_size:
    :return:
    """
    shape = lung_img.shape
    output = []
    lung_img = lung_img + 1024
    for i in range(len(coords)):
        patch = lung_img[coords[i][0] - 18: coords[i][0] + 18,
                coords[i][1] - 18: coords[i][1] + 18,
                coords[i][2] - 18: coords[i][2] + 18]
        output.append(patch)

    return output

def get_point(shape):
    """
    Sample a random point from the image and return the coordinates.
    :param shape:
    :return:
    """
    x = np.random.randint(50, shape[2] - 50)
    y = np.random.randint(50, shape[1] - 50)
    z = np.random.randint(20, shape[0] - 20)
    return np.asarray([z, y, x])

def create_data(path, train_csv_path):
    """
    This function reads the training csv file which contains the CT Scan names and
    location of each nodule. It cuts many voxels around a nodule and takes random points as
    negative samples. The voxels are dumped using pickle. It is to be noted that the voxels are
    cut from original Scans and not the masked CT Scans generated while creating candidate
    regions.
    :param path:
    :param train_csv_path:
    :return:
    """
    coords, trainY = [], []

    with open(train_csv_path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            row = line.split(',')

            if os.path.isfile(path + row[0] + '.mhd') == False:
                continue

            lung_img = sitk.GetArrayFromImage(sitk.ReadImage(path + row[0] + '.mhd'))

            for i in range(-5, 5, 3):
                for j in range(-5, 5, 3):
                    for k in range(-2, 3, 2):
                        coords.append([int(row[3]) + k, int(row[2]) + j, int(row[1]) + i])
                        trainY.append(True)

            for i in range(60):
                coords.append(get_point(lung_img.shape))
                trainY.append(False)

            trainX = get_patch_from_list(lung_img, coords)

            pickle.dump(np.asarray(trainX), open('traindata_' + str(counter) + '_Xtrain.p', 'wb'))
            pickle.dump(np.asarray(trainY, dtype=bool), open('traindata_' + str(counter) + '_Ytrain.p', 'wb'))

            counter = counter + 1
            coords, trainY = [], []

