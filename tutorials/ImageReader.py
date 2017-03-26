import numpy as np # linear algebra
import dicom
import os
import SimpleITK as sitk

def load_scan(path):
    """
    Load all slices and metadata from dicom images in path to patient
    :param path: path to a patient's scans
    :return: the metadata for the image slices
    """
    slices = [dicom.read_file(os.path.join(path, filename)) for filename in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_itk(filename):
    """
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    :param filename: file containing image from LUNA16 dataset
    :return:
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing