import os
import nibabel
from PIL import Image
import numpy as np
import SimpleITK as sitk

def search_file_in_folder_list(folder_list, file_name):
    """ search a file with a part of name in a list of folders
    input:
        folder_list: a list of folders
        file_name:   a substring of a file
    output:
        the full file name
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name


def load_nifty_volume_as_array(filename, with_spacing=False):
    """Read a nifty image and return data array
    input shape [W, H, D]
    output shape [D, H, W]
    """
    img = nibabel.load(filename)
    data = img.get_data()
    shape = data.shape
    if(len(shape) == 4):
        assert(shape[3] == 1)
        data = np.reshape(data, shape[:-1])
    data = np.transpose(data, [2,1,0])
    if(with_spacing):
        spacing = img.header.get_zooms()
        spacing = [spacing[2], spacing[1], spacing[0]]
        return data, spacing
    return data, None

def load_image_as_array(image_name, with_spacing = True):
    if (image_name.endswith(".nii.gz") or image_name.endswith(".mha")):
        image, spacing = load_nifty_volume_as_array(image_name, with_spacing)
        image = np.asarray([image], np.float32)
        image = np.transpose(image, [1, 2, 3, 0]) # [D, H, W, C]
        return image, spacing
    elif(image_name.endswith(".jpg") or image_name.endswith(".png")):
        image = np.asarray(Image.open(image_name), np.float32)
        if(len(image.shape) == 3):
            image = np.expand_dims(image, axis = 0) # [D, H, W, C]
        else:
            image = np.expand_dims(image, axis = 0)
            image = np.expand_dims(image, axis = -1)
        return image, (1.0, 1.0, 1.0)
    else:
        raise ValueError("unsupported image format")

def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)
