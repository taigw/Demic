import os
from PIL import Image
import numpy as np

def convert_JCRT_images(input_dir, output_dir):
    file_names = os.listdir(input_dir)
    file_names = [item for item in file_names if item[:2] == "JP"]
    print("image number: {0:}".format(len(file_names)))

    for file_name in file_names:
        input_file_name = os.path.join(input_dir,file_name)
        output_file_name = file_name[:-3] + "png"
        output_file_name = os.path.join(output_dir, output_file_name)

        shape = (2048, 2048) # matrix size
        dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)

        # Reading.
        fid = open(input_file_name, 'rb')
        data = np.fromfile(fid, dtype)
        data = data.reshape(shape)

        # Rescale intensity to 0-255
        data[data > 4096] = 4096
        data = data * 255.0 / 4096 #data.max()
        data = np.asarray(data, np.uint8)
        
        # Reshape to 256x256
        
        img = Image.fromarray(data)
        img = img.resize((256, 256), Image.BILINEAR)
        img.save(output_file_name)
        print(file_name, data.min(), data.max(), data.mean(), data.std())

def convert_JCRT_labels(input_dir, output_dir):
    sub_folders = ['fold1/masks/heart','fold2/masks/heart']
    for sub_folder in sub_folders:
        input_label_dir = os.path.join(input_dir, sub_folder)
        file_names = os.listdir(input_label_dir)
        file_names = [item for item in file_names if item[:2] == "JP"]
        for file_name in file_names:
            input_full_name = os.path.join(input_label_dir, file_name)
            output_full_name = file_name[:-4] + "_lab.png"
            output_full_name = os.path.join(output_dir, output_full_name)
            img = Image.open(input_full_name)
            img = img.resize((256, 256), Image.NEAREST)
            img.save(output_full_name)
            print(file_name[:-4])

if __name__ == "__main__":
    input_image_dir = "D:/Documents/data/JSRT/All247images"
    output_image_dir = "D:/Documents/data/JSRT/image"
    convert_JCRT_images(input_image_dir, output_image_dir)

    input_label_dir = "D:/Documents/data/JSRT/scratch"
    output_label_dir = "D:/Documents/data/JSRT/label"
    convert_JCRT_labels(input_label_dir, output_label_dir)