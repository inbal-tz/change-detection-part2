from dataset import idd_lite
import os
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

PATH_A = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\images\test\A_whole'
PATH_B = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\images\test\B_whole'
PATH_LABEL = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\labels\test_whole'
OUTPUT_A = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\images\test\A'
OUTPUT_B = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\images\test\B'
OUTPUT_LABEL = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\labels\test'
NUM_WORKERS = 0
BATCH_SIZE = 1
EXTENSIONS = ['.jpg', '.png']
CROPPED_SIZE = 128
#
# def load_image(file):
#     return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def crop(path, height, width):
    result = []
    im = Image.open(path)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight, height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            result.append(im.crop(box))
    return result


def split_image (input_path, output_path):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input_path)) for f in fn if
                   is_image(f)]
    for image in file_names:
        for k, piece in enumerate(crop(image, CROPPED_SIZE, CROPPED_SIZE)):
            img = Image.new('RGB', (CROPPED_SIZE, CROPPED_SIZE), 255)
            img.paste(piece)
            path = os.path.join(output_path, os.path.basename(image).split(".")[0] + "_" + str(k) + ".jpg")
            img.save(path)


split_image(PATH_A, OUTPUT_A)
split_image(PATH_B, OUTPUT_B)
split_image(PATH_LABEL, OUTPUT_LABEL)
