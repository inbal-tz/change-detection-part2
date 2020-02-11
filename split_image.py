from dataset import idd_lite
import os
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

PATH_A = r'D:\Users Data\inbal.tlGIP\Desktop\ChangeDetectionDataset\images\test\A'
PATH_B = r'D:\Users Data\inbal.tlGIP\Desktop\ChangeDetectionDataset\images\test\B'
PATH_LABEL = r'D:\Users Data\inbal.tlGIP\Desktop\ChangeDetectionDataset\labels\train'
OUTPUT_A = r'D:\Users Data\inbal.tlGIP\Desktop\part b\images\test\croped A'
OUTPUT_B = r'D:\Users Data\inbal.tlGIP\Desktop\part b\images\test\croped B'
OUTPUT_LABEL = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\trainCropped'
NUM_WORKERS = 0
BATCH_SIZE = 1
EXTENSIONS = ['.jpg', '.png']
CROPPED_SIZE = 64
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


# split_image(PATH_A, OUTPUT_A)
# split_image(PATH_B, OUTPUT_B)
split_image(PATH_LABEL, OUTPUT_LABEL)
