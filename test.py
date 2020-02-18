from PIL import Image
import os
input_path = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\trainCropped'
import torch

def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.png', '.tiff'])


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

# def calcPrecentWhiteVsProbabilityForChange():
#     from collections import Counter
#     import matplotlib.pyplot as plt
#     white_precents=[]
#     probabilities=[]
#     dataset_test = idd_lite(DATA_ROOT, co_transform_val, 'test')
#     loader_test = DataLoader(dataset_test, num_workers=NUM_WORKERS, batch_size=1, shuffle=True)
#     for step, (images, images1, labels, filename) in enumerate(loader_test):
#         inputs = images.to(device)
#         inputs1 = images1.to(device)  # ChangedByUs
#         targets = Image.open(os.path.join(r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\testCropped', filename[0]+'.jpg'))
#         targets = targets.convert('1')
#         targets = list(targets.getdata())
#         if 0 not in Counter(targets):
#             white_precent = 1
#         elif 255 not in Counter(targets):
#             white_precent = 0
#         else:
#             white_precent = Counter(targets)[255]/(Counter(targets)[0]+Counter(targets)[255])
#         try:
#             output, GAP = model([inputs.to(device), inputs1.to(device)], only_encode=ENCODER_ONLY)
#         except:
#             print(filename[0])
#         white_precents.append(white_precent)
#         probabilities.append(torch.nn.Softmax()(output[0])[1].data.tolist())
#     plt.scatter(white_precents, probabilities)
#     plt.xlabel("precent change in image")
#     plt.ylabel("probability that a change occured")
#     plt.show()
#

def label2pixel(label):
    im = Image.open(label)
    im = im.convert('1')
    im = list(im.getdata())
    pixel_val = max(im) #255 is white. 0 is black. if label has white- there is a change
    return pixel_val

device = 'cuda'
input_path_model = r'D:\Users Data\inbal.tlGIP\Desktop\part b\output_test'
input_path_label = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\test'
FP=0
TP=0
FN=0
TN=0
file_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input_path_model)) for f in fn if is_image(f)]
for image in file_names:
    output = label2pixel(image)
    label = Image.open(os.path.join(input_path_label,os.path.basename(image).split(".")[0] + ".jpg"))
    label = label.convert('1')
    label = list(label.getdata())[0]
    if label == 0 and output == 0:
        TN += 1
    if label == 0 and output == 255:
        FP += 1
    if label == 255 and output == 0:
        FN += 1
    if label == 255 and output == 255:
        TP += 1
print('FP: ', FP, 'FN: ', FN, 'TP: ', TP, 'TN: ', TN)
if TP + FN > 0 and TP + FP > 0:
    print('recall= ', TP / (TP + FN), 'precision= ', TP / (TP + FP))
    # label = label.to(device)
    # label[label < 128] = 0  # ChangedByUs
    # label[label >= 128] = 1  # ChangedByUs
    # label = torch.LongTensor([target.cpu().numpy().flatten()[0] for target in label])[0]
    x=1


# label = label.to(device)
