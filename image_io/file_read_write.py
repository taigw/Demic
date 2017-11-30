
import numpy as np
import nibabel


def load_nifty_volume_as_array(filename, with_spacing):
    """Read a nifty image and return data array
    input shape [W, H, D]
    output shape [D, H, W]
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_spacing):
        spacing = img.header.get_zooms()
        spacing = [spacing[2], spacing[1], spacing[0]]
        return data, spacing
    return data


def save_array_as_nifty_volume(data, filename):
    """Write a numpy array as nifty image
        numpy data shape [D, H, W]
        nifty image shape [W, H, D]
        """
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)
