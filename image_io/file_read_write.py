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


def load_nifty_volume_as_4d_array(filename):
    """Read a nifty image and return a dictionay storing data array, spacing and direction
    output['data_array'] 4d array with shape [D, H, W, C]
    output['spacing']    a list of spacing in z, y, x axis 
    output['direction']  a 3x3 matrix for direction
    """
    img_obj = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    direction = img_obj.GetDirection()
    shape = data_array.shape
    if(len(shape) == 4):
        assert(shape[3] == 1) 
    elif(len(shape) == 3):
        data_array = np.expand_dims(data_array, axis = -1)
    else:
        raise ValueError("unsupported image dim: {0:}".format(len(shape)))
    output = {}
    output['data_array'] = data_array
    output['spacing']    = (spacing[2], spacing[1], spacing[0])
    output['direction']  = direction
    return output

def load_rgb_image_as_4d_array(filename):
    image = np.asarray(Image.open(filename))
    if(len(image.shape) == 3):
            image = np.expand_dims(image, axis = 0) # [D, H, W, C]
    else:
        image = np.expand_dims(image, axis = 0)
        image = np.expand_dims(image, axis = -1)
    output = {}
    output['data_array'] = image
    output['spacing']    = (1.0, 1.0, 1.0)
    output['direction']  = None
    return output

def load_image_as_4d_array(image_name):
    """
    return a 4D array with shape [D, H, W, C]
    """
    if (image_name.endswith(".nii.gz") or image_name.endswith(".mha")):
        image_dict = load_nifty_volume_as_4d_array(image_name)
    elif(image_name.endswith(".jpg") or image_name.endswith(".png")):
        image_dict = load_rgb_image_as_4d_array(image_name)
    else:
        raise ValueError("unsupported image format")
    return image_dict

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

