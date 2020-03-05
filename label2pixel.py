from PIL import Image
import os

INPUT_PATH = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\labels\test_cropped'
OUTPUT_PATH = r'C:\Users\inbal.tlgip\Desktop\ChangeDetectionDataset\Real\subset\labels\test'
EXTENSIONS = ['.jpg', '.png']


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def label2pixel(label):
    im = Image.open(label)
    im = im.convert('1')
    im = list(im.getdata())
    pixel_val = max(im) #255 is white. 0 is black. if label has white- there is a change
    pixel = Image.new('1', (1, 1), pixel_val)
    return pixel


file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(INPUT_PATH)) for f in fn if is_image(f)]

for image in file_names:
    pixel = label2pixel(image)
    path = os.path.join(OUTPUT_PATH, os.path.basename(image))
    pixel.save(path)

