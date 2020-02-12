from PIL import Image
import os
input_path = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\trainCropped'


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


def get_balanced_dataBase(dataBase_length):
    file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input_path)) for f in fn if is_image(f)]
    if len(file_names) <= dataBase_length:
        return file_names
    balanced_data_base = []
    not_added=[]
    with_change = 0
    no_change = 0
    for image in file_names:
        if len(balanced_data_base) >= dataBase_length:
            break
        im = Image.open(image)
        im = im.convert('1')
        im = list(im.getdata())
        pixel_val = max(im)
        if pixel_val == 255 and with_change <= dataBase_length/2:
            balanced_data_base.append(image)
            with_change += 1
        elif pixel_val == 0 and no_change <= dataBase_length/2:
            balanced_data_base.append(image)
            no_change += 1
        else:
            not_added.append(image)
    if len(balanced_data_base) < dataBase_length:
        balanced_data_base += not_added[0:dataBase_length-len(balanced_data_base)]
    return balanced_data_base


res = get_balanced_dataBase(10000)
x=1