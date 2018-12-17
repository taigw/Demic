import os
import random
def data_split():
    image_dir = "D:/Documents/data/JSRT/image"
    random.seed(0)
    file_names = os.listdir(image_dir)
    patient_names = [item[:-4] for item in file_names if item[:2] == "JP"]
    print("image number", len(patient_names))

    # reorder the patient names randomly
    random.shuffle(patient_names)
    train_names = patient_names[:180]
    valid_names = patient_names[180:200]
    test_names  = patient_names[200:]

    with open("config/train_names.txt", 'w') as f:
        for name in train_names:
            f.write("{0:}\n".format(name))
    f.close()

    with open("config/valid_names.txt", 'w') as f:
        for name in valid_names:
            f.write("{0:}\n".format(name))
    f.close()

    with open("config/test_names.txt", 'w') as f:
        for name in test_names:
            f.write("{0:}\n".format(name))
    f.close()
    
    print("number of training images : {0:}".format(len(train_names)))
    print("number of validation images : {0:}".format(len(valid_names)))
    print("number of testing images : {0:}".format(len(test_names)))
if __name__ == "__main__":
    data_split()