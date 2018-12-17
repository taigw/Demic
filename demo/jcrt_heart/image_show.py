from PIL import Image
import matplotlib.pyplot  as plt
from Demic.util.image_visualize import add_countor

def show_image_with_segmentation(image_name, label_name):
    img = Image.open(image_name)
    lab = Image.open(label_name)
    img_show = img.convert('RGB')
    img_show = add_countor(img_show, lab)
    plt.imshow(img_show)
    plt.show()


if __name__ == "__main__":
    image_name = "D:/Documents/data/JSRT/image/JPCLN001.png"
    label_name = "D:/Documents/data/JSRT/label/JPCLN001_lab.png"
    show_image_with_segmentation(image_name, label_name)